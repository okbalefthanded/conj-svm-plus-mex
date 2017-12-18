//
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include <memory.h>
#include <time.h>

// #include <sys/param.h>
// #include <sys/times.h>
//
#include "svm.h"

// common.h
//
#include "mex.h"
//
int libsvm_version = LIBSVM_VERSION;

typedef double Qfloat;
typedef signed char schar;
#ifndef min
template <class T>
inline T min(T x, T y)
{
  return (x < y) ? x : y;
}
#endif

#ifndef max
template <class T>
inline T max(T x, T y)
{
  return (x > y) ? x : y;
}
#endif

template <class T>
inline void swap(T &x, T &y)
{
  T t = x;
  x = y;
  y = t;
}
template <class S, class T>
inline void clone(T *&dst, S *src, int n)
{
  dst = new T[n];
  memcpy((void *)dst, (void *)src, sizeof(T) * n);
}
inline double powi(double base, int times)
{
  double tmp = base, ret = 1.0;

  for (int t = times; t > 0; t /= 2)
  {
    if (t % 2 == 1)
      ret *= tmp;
    tmp = tmp * tmp;
  }
  return ret;
}

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

enum
{
  BETA_I_BETA_J,
  ALPHA_I_ALPHA_J,
  ALPHA_I_ALPHA_J_BETA_K
};

static void print_string_stdout(const char *s)
{
  fputs(s, stdout);
  fflush(stdout);
}

static void (*svm_print_string)(const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt, ...)
{
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);
  (*svm_print_string)(buf);
}
#else
static void info(const char *fmt, ...)
{
}
#endif
//
// Kernel Cache (cache.h)
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
  Cache(int l, long int size);
  ~Cache();

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  int get_data(const int index, Qfloat **data, int len);
  void swap_index(int i, int j);

private:
  int l;
  long int size;
  struct head_t
  {
    head_t *prev, *next; // a circular list
    Qfloat *data;
    int len; // data[0,len) is cached in this entry
  };

  head_t *head;
  head_t lru_head;
  void lru_delete(head_t *h);
  void lru_insert(head_t *h);
};
// Cache.cpp
Cache::Cache(int l_, long int size_) : l(l_), size(size_)
{
  head = (head_t *)calloc(l, sizeof(head_t)); // initialized to 0
  size /= sizeof(Qfloat);
  size -= l * sizeof(head_t) / sizeof(Qfloat);
  size = max(size, 2 * (long int)l); // cache must be large enough for two columns
  lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
  for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
    free(h->data);
  free(head);
}

void Cache::lru_delete(head_t *h)
{
  // delete from current location
  h->prev->next = h->next;
  h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
  // insert to last position
  h->next = &lru_head;
  h->prev = lru_head.prev;
  h->prev->next = h;
  h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
  head_t *h = &head[index];
  if (h->len)
    lru_delete(h);
  int more = len - h->len;

  if (more > 0)
  {
    // free old space
    while (size < more)
    {
      head_t *old = lru_head.next;
      lru_delete(old);
      free(old->data);
      size += old->len;
      old->data = 0;
      old->len = 0;
    }

    // allocate new space
    h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat) * len);
    size -= more;
    swap(h->len, len);
  }

  lru_insert(h);
  *data = h->data;
  return len;
}

void Cache::swap_index(int i, int j)
{
  if (i == j)
    return;

  if (head[i].len)
    lru_delete(&head[i]);
  if (head[j].len)
    lru_delete(&head[j]);
  swap(head[i].data, head[j].data);
  swap(head[i].len, head[j].len);
  if (head[i].len)
    lru_insert(&head[i]);
  if (head[j].len)
    lru_insert(&head[j]);

  if (i > j)
    swap(i, j);
  for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
  {
    if (h->len > i)
    {
      if (h->len > j)
        swap(h->data[i], h->data[j]);
      else
      {
        // give up
        lru_delete(h);
        free(h->data);
        size += h->len;
        h->data = 0;
        h->len = 0;
      }
    }
  }
}
// kernel.h
//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix
{
public:
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual Qfloat *get_QD() const = 0;
  virtual void swap_index(int i, int j) const = 0;
  virtual ~QMatrix() {}
};

class Kernel : public QMatrix
{
public:
  Kernel(int l, svm_node *const *x, const svm_parameter &param);
  virtual ~Kernel();

  static double k_function(const svm_node *x, const svm_node *y,
                           const svm_parameter &param);
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual Qfloat *get_QD() const = 0;
  virtual void swap_index(int i, int j) const // no so const...
  {
    swap(x[i], x[j]);
    if (x_square)
      swap(x_square[i], x_square[j]);
  }

protected:
  double (Kernel::*kernel_function)(int i, int j) const;

private:
  const svm_node **x;
  double *x_square;

  // svm_parameter
  const int kernel_type;
  const int degree;
  const double gamma;
  const double coef0;

  static double dot(const svm_node *px, const svm_node *py);
  double kernel_linear(int i, int j) const
  {
    return dot(x[i], x[j]);
  }
  double kernel_poly(int i, int j) const
  {
    return powi(gamma * dot(x[i], x[j]) + coef0, degree);
  }
  double kernel_rbf(int i, int j) const
  {
    return exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
  }
  double kernel_sigmoid(int i, int j) const
  {
    return tanh(gamma * dot(x[i], x[j]) + coef0);
  }
  double kernel_precomputed(int i, int j) const
  {
    return x[i][(int)(x[j][0].value)].value;
  }
};

// Kernel.cpp
Kernel::Kernel(int l, svm_node *const *x_, const svm_parameter &param)
    : kernel_type(param.kernel_type), degree(param.degree),
      gamma(param.gamma), coef0(param.coef0)
{
  switch (kernel_type)
  {
  case LINEAR:
    kernel_function = &Kernel::kernel_linear;
    break;
  case POLY:
    kernel_function = &Kernel::kernel_poly;
    break;
  case RBF:
    kernel_function = &Kernel::kernel_rbf;
    break;
  case SIGMOID:
    kernel_function = &Kernel::kernel_sigmoid;
    break;
  case PRECOMPUTED:
    kernel_function = &Kernel::kernel_precomputed;
    break;
  }

  clone(x, x_, l);

  if (kernel_type == RBF)
  {
    x_square = new double[l];
    for (int i = 0; i < l; i++)
      x_square[i] = dot(x[i], x[i]);
  }
  else
    x_square = 0;
}

Kernel::~Kernel()
{
  delete[] x;
  delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
  // info("inside kernel::dot\n");
  double sum = 0;
  while (px->index != -1 && py->index != -1)
  {
    if (px->index == py->index)
    {
      sum += px->value * py->value;
      ++px;
      ++py;
    }
    else
    {
      if (px->index > py->index)
        ++py;
      else
        ++px;
    }
  }
  return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
                          const svm_parameter &param)
{
  switch (param.kernel_type)
  {
  case LINEAR:
    return dot(x, y);
  case POLY:
    return powi(param.gamma * dot(x, y) + param.coef0, param.degree);
  case RBF:
  {
    double sum = 0;
    while (x->index != -1 && y->index != -1)
    {
      if (x->index == y->index)
      {
        double d = x->value - y->value;
        sum += d * d;
        ++x;
        ++y;
      }
      else
      {
        if (x->index > y->index)
        {
          sum += y->value * y->value;
          ++y;
        }
        else
        {
          sum += x->value * x->value;
          ++x;
        }
      }
    }

    while (x->index != -1)
    {
      sum += x->value * x->value;
      ++x;
    }

    while (y->index != -1)
    {
      sum += y->value * y->value;
      ++y;
    }

    return exp(-param.gamma * sum);
  }
  case SIGMOID:
    return tanh(param.gamma * dot(x, y) + param.coef0);
  case PRECOMPUTED: //x: test (validation), y: SV
    return x[(int)(y->value)].value;
  default:
    return 0; // Unreachable
  }
}

// solve_linear_systems.c
#define TINY 1.0e-20
void lubksb(double **a, int n, int *indx, double b[])
{
  int i, ii = -1, ip, j;
  double sum;

  for (i = 0; i < n; i++)
  {
    ip = indx[i];
    sum = b[ip];
    b[ip] = b[i];
    b[i] = sum;
    if (ii != -1)
      for (j = ii; j <= i - 1; j++)
        sum -= a[i][j] * b[j];
    else if (sum)
      ii = i;
    b[i] = sum;
  }

  for (i = n - 1; i >= 0; i--)
  {
    sum = b[i];
    for (j = i + 1; j < n; j++)
      sum -= a[i][j] * b[j];
    b[i] = sum / a[i][i];
  }
}

int ludcmp(double **a, int n, int *indx, int *d)
{
  int i, imax, j, k;
  double big, dum, sum, temp;
  double *vv;

  vv = new double[n];
  *d = 1;
  for (i = 0; i < n; i++)
  {
    big = 0.0;
    for (j = 0; j < n; j++)
      if ((temp = fabs(a[i][j])) > big)
        big = temp;
    if (big == 0.0)
      return 1;
    vv[i] = 1.0 / big;
  }
  for (j = 0; j < n; j++)
  {
    for (i = 0; i < j; i++)
    {
      sum = a[i][j];
      for (k = 0; k < i; k++)
        sum -= a[i][k] * a[k][j];
      a[i][j] = sum;
    }
    big = 0.0;
    for (i = j; i < n; i++)
    {
      sum = a[i][j];
      for (k = 0; k < j; k++)
        sum -= a[i][k] * a[k][j];
      a[i][j] = sum;
      if ((dum = vv[i] * fabs(sum)) >= big)
      {
        big = dum;
        imax = i;
      }
    }
    if (j != imax)
    {
      for (k = 0; k < n; k++)
      {
        dum = a[imax][k];
        a[imax][k] = a[j][k];
        a[j][k] = dum;
      }
      *d = -(*d);
      vv[imax] = vv[j];
    }
    indx[j] = imax;

    if (a[j][j] == 0.0)
      a[j][j] = TINY;
    if (j != n - 1)
    {
      dum = 1.0 / (a[j][j]);
      for (i = j + 1; i < n; i++)
        a[i][j] *= dum;
    }
  }

  delete[] vv;
  return 0;
}

int solve_linear_system(double **a, double *b, int n)
{
  int *indx, result;
  int d;

  indx = new int[n];
  result = ludcmp(a, n, indx, &d);

  if (result)
  {
    delete[] indx;
    return 1;
  }
  lubksb(a, n, indx, b);

  delete[] indx;
  return 0;
}

// Solver.h
// An SMO and conjugate SMO algorithms for SVM
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver
{

public:
  // common.c
  struct SolutionInfo
  {
    double obj;
    double rho;
    double rho_star;
    double upper_bound_p;
    double upper_bound_n;
    double upper_bound_p_star;
    double upper_bound_n_star;
    double r; // for Solver_NU
  };

  Solver(int optimizer_ = -1)
  {
    if (optimizer_ == -1)
      conjugate = false;
    else
    {
      conjugate = true;
      max_depth = optimizer_;
      A = new double *[max_depth + 1];
      for (int i = 0; i <= max_depth; i++)
        A[i] = new double[max_depth + 1];
      b = new double[max_depth + 1];
    }
    
    info("conjugate=%d\n", conjugate);
    // fprintf(stdout, "conjugate=%d\n", conjugate);
    // fflush(stdout);

    sizeof_double = sizeof(double);
    sizeof_char = sizeof(char);
    sizeof_int = sizeof(int);
  };
  virtual ~Solver()
  {
    if (conjugate)
    {
      for (int i = 0; i <= max_depth; i++)
        delete[] A[i];
      delete[] A;
      delete[] b;
    }
  };

  void Solve(int l, const QMatrix &Q, const double *p_, const schar *y_,
             double *alpha_, double Cp, double Cn, double eps,
             SolutionInfo *si, int shrinking);

  void Solve_cg(int l, const QMatrix &Q, const double *p_, const schar *y_,
                double *alpha_, double Cp, double Cn, double eps,
                SolutionInfo *si, int shrinking);

protected:
  int active_size;
  schar *y;
  double *G; // gradient of objective function
  double **G_cg;
  enum
  {
    LOWER_BOUND,
    UPPER_BOUND,
    FREE
  };
  char *alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE
  char **alpha_status_cg;
  double *alpha;
  double **alpha_cg;
  const QMatrix *Q;
  const Qfloat *QD;
  double eps;
  double Cp, Cn;
  double *p;
  int *active_set; // maps permuted indices to the original ones
  double *G_bar;   // gradient, if we treat free variables as 0
  double **G_bar_cg;
  int l;
  bool unshrink; // XXX
  bool conjugate;
  int *work;
  int max_depth;

  double get_C(int i)
  {
    return (y[i] > 0) ? Cp : Cn;
  }
  void update_alpha_status(int i)
  {
    if (alpha[i] >= get_C(i) - 1e-12)
      alpha_status[i] = UPPER_BOUND;
    else if (alpha[i] <= 1e-12)
      alpha_status[i] = LOWER_BOUND;
    else
      alpha_status[i] = FREE;
  }
  void update_alpha_status_cg(int i, int depth)
  {
    if (alpha_cg[depth][i] >= get_C(i) - 1e-12)
      alpha_status_cg[depth][i] = UPPER_BOUND;
    else if (alpha_cg[depth][i] <= 1e-12)
      alpha_status_cg[depth][i] = LOWER_BOUND;
    else
      alpha_status_cg[depth][i] = FREE;
  }
  bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
  bool is_upper_bound_cg(int i, int depth) { return alpha_status_cg[depth][i] == UPPER_BOUND; }
  bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
  bool is_lower_bound_cg(int i, int depth) { return alpha_status_cg[depth][i] == LOWER_BOUND; }
  bool is_free(int i) { return alpha_status[i] == FREE; }
  bool is_free_cg(int i, int depth) { return alpha_status_cg[depth][i] == FREE; }
  void swap_index(int i, int j);
  void reconstruct_gradient();
  int generate_direction(double *u, int n, bool new_working_set);
  virtual int wss_first_order(double *u, double &lambda_star);
  virtual int select_working_set_hmg(double *u, double &lambda_star, double &gain);
  virtual int select_working_set_hmg2(double *u, double &lambda_star, double &gain);
  virtual int select_working_set(int &i, int &j);
  virtual double calculate_rho();
  virtual double calculate_rho_cg();
  virtual void do_shrinking();
  virtual bool do_shrinking_cg();

private:
  bool bumped();
  bool be_shrunk(int i, double Gmax1, double Gmax2);
  bool be_shrunk_cg(int i, double Gmax1, double Gmax2);
  void generate_direction3(double *u, bool new_working_set);
  void generate_direction4(double *u, bool new_working_set);
  int generate_direction_general(double *u, int n, bool new_working_set);
  int select_working_set_first_order();
  void compute_step_first_order(double *u, double &lambda_star);
  int select_working_set_incrementally(double *u, double &lambda_star, double &gain);
  bool feasible_direction(double *u);
  int iter;
  int curr_depth;
  double **A;
  double *b;
  int sizeof_double;
  int sizeof_int;
  int sizeof_char;
  bool *active;
};

// Solver.cpp
bool Solver::bumped()
{
  for (int i = 0; i < curr_depth + 2; i++)
    if (is_upper_bound_cg(work[i], 0) || is_lower_bound_cg(work[i], 0))
      return true;
  return false;
}

void Solver::swap_index(int i, int j)
{
  Q->swap_index(i, j);
  swap(y[i], y[j]);
  swap(p[i], p[j]);
  swap(active_set[i], active_set[j]);

  if (!conjugate)
  {
    swap(G[i], G[j]);
    swap(alpha_status[i], alpha_status[j]);
    swap(alpha[i], alpha[j]);
    swap(G_bar[i], G_bar[j]);
  }
  else
  {
    int k;
    for (k = 0; k < curr_depth + 1; k++)
    {
      swap(G_cg[k][i], G_cg[k][j]);
      swap(alpha_status_cg[k][i], alpha_status_cg[k][j]);
      swap(alpha_cg[k][i], alpha_cg[k][j]);
      swap(G_bar_cg[k][i], G_bar_cg[k][j]);
    }
    for (k = 0; k < curr_depth + 2; k++)
    {
      if (i == work[k])
        work[k] = -1;
      else if (j == work[k])
        work[k] = i;
    }
    active[i] = active[j];
    active[j] = false;
  }
}

void Solver::reconstruct_gradient()
{
  // reconstruct inactive elements of G and G_old from G_bar, G_bar_old and free variables
  int i, j;
  int nr_free = 0;

  if (!conjugate)
  {

    if (active_size == l)
      return;

    for (j = active_size; j < l; j++)
      G[j] = G_bar[j] + p[j];
    for (j = 0; j < active_size; j++)
    {
      if (is_free(j))
        nr_free++;
    }

    if (2 * nr_free < active_size)
      info("\nWarning: using -h 0 may be faster\n");

    if (nr_free * l > 2 * active_size * (l - active_size))
      for (i = active_size; i < l; i++)
      {
        const Qfloat *Q_i = Q->get_Q(i, active_size);
        for (j = 0; j < active_size; j++)
          if (is_free(j))
            G[i] += alpha[j] * Q_i[j];
      }
    else
      for (i = 0; i < active_size; i++)
        if (is_free(i))
        {
          const Qfloat *Q_i = Q->get_Q(i, l);
          double alpha_i = alpha[i];

          if (is_free(i))
            for (j = active_size; j < l; j++)
              G[j] += alpha_i * Q_i[j];
        }
  }
  else
  {

    if (active_size == l)
      return;

    //fprintf(stdout,"reconstructing gradient\n");
    //        fflush(stdout);

    for (int k = 0; k < curr_depth + 1; k++)
    {
      for (j = active_size; j < l; j++)
        G_cg[k][j] = G_bar_cg[k][j] + p[j];

      nr_free = 0;
      for (j = 0; j < active_size; j++)
        if (is_free_cg(j, k))
          nr_free++;

      if (nr_free * l > 2 * active_size * (l - active_size))
        for (i = active_size; i < l; i++)
        {
          const Qfloat *Q_i = Q->get_Q(i, active_size);
          for (j = 0; j < active_size; j++)
            if (is_free_cg(j, k))
              G_cg[k][i] += alpha_cg[k][j] * Q_i[j];
        }

      else
        for (i = 0; i < active_size; i++)
          if (is_free_cg(i, k))
          {
            const Qfloat *Q_i = Q->get_Q(i, l);
            double alpha_i = alpha_cg[k][i];

            if (is_free_cg(i, k))
              for (j = active_size; j < l; j++)
                G_cg[k][j] += alpha_i * Q_i[j];
          }
    }
  }
}

void Solver::Solve_cg(int l, const QMatrix &Q, const double *p_, const schar *y_,
                      double *alpha_, double Cp, double Cn, double eps,
                      SolutionInfo *si, int shrinking)
{
  int i, j;
  double delta_alpha_i, C_i;

  this->l = l;
  this->Q = &Q;
  QD = Q.get_QD();
  clone(p, p_, l);
  clone(y, y_, l);
  this->Cp = Cp;
  this->Cn = Cn;
  this->eps = eps;
  unshrink = false;
  curr_depth = -1;
  active_size = l;

  work = new int[max_depth + 2];
  for (i = 0; i < max_depth + 2; i++)
    work[i] = -1;

  active = new bool[l];
  for (i = 0; i < l; i++)
    active[i] = false;

  // initialize alpha's
  alpha_cg = new double *[max_depth + 1];
  for (i = 0; i < max_depth + 1; i++)
    clone(alpha_cg[i], alpha_, l);

  // initialize alpha_status
  alpha_status_cg = new char *[max_depth + 1];
  for (i = 0; i < max_depth + 1; i++)
  {
    alpha_status_cg[i] = new char[l];
    for (j = 0; j < l; j++)
    {
      update_alpha_status_cg(j, i);
    }
  }

  // initialize active set (for shrinking)
  active_set = new int[l];
  for (i = 0; i < l; i++)
    active_set[i] = i;

  // initialize gradient
  G_cg = new double *[max_depth + 1];
  G_bar_cg = new double *[max_depth + 1];
  G_cg[0] = new double[l];
  G_bar_cg[0] = new double[l];

  for (i = 0; i < l; i++)
  {
    G_cg[0][i] = p[i];
    G_bar_cg[0][i] = 0;
  }

  for (i = 0; i < l; i++)
    if (!is_lower_bound_cg(i, 0))
    {
      const Qfloat *Q_i = Q.get_Q(i, l);
      double alpha_i = alpha_cg[0][i];
      int j;
      for (j = 0; j < l; j++)
        G_cg[0][j] += alpha_i * Q_i[j];
      if (is_upper_bound_cg(i, 0))
      {
        C_i = get_C(i);
        for (j = 0; j < l; j++)
          G_bar_cg[0][j] += C_i * Q_i[j];
      }
    }

  for (i = 1; i < max_depth + 1; i++)
  {
    clone(G_cg[i], G_cg[0], l);
    clone(G_bar_cg[i], G_bar_cg[0], l);
  }

  // optimization step

  iter = 0;
  int counter = min(l, 1000) + 1;
  double *u, *tmp_u;
  double lambda_star, tmp_lambda, gain_hmg, gain2;
  bool corner = false, upper_bound_i, bumped = false;
  double alpha_i;
  Qfloat *Q_i;
  double *tmp_alpha, *tmp_G, *tmp_G_bar;
  char *tmp_status;
  int work_i, prev_depth;
  int *tmp_work, tmp_depth, tmp_working_set_size, working_set_size;
  double old_alpha[2];
  bool old_upper_bound[2], done_shrinking = false;
  int last_updated_G_bar = -1;
  int n_conjugate = 0, next_depth;

  u = new double[max_depth + 2];
  tmp_u = new double[max_depth + 2];
  tmp_work = new int[max_depth + 2];

  while (1)
  {

    // do shrinking
    ///////////////////////////////////////
    if (--counter == 0)
    {
      counter = min(l, 1000);
      if (shrinking)
        done_shrinking = do_shrinking_cg();
    }
    ///////////////////////////////////////

    // do working set selection
    ////////////////////////////////////////////////////////////

    prev_depth = curr_depth;

    if (corner || iter == 0)
    {
      if (wss_first_order(u, lambda_star) != 0)
      {
        reconstruct_gradient();
        active_size = l;
        if (wss_first_order(u, lambda_star) != 0)
          break;
        else
          counter = 1;
      }
    }
    else
    {
      if (curr_depth >= 0 && max_depth > 0)
      {
        if (select_working_set_hmg(u, lambda_star, gain_hmg) != 0)
        {
          memcpy(tmp_work, work, sizeof_int * (curr_depth + 2));
          tmp_depth = curr_depth;
          for (i = 0; i < curr_depth + 2; i++)
          {
            active[work[i]] = false;
            work[i] = -1;
          }
          curr_depth = -1;
          if (select_working_set_hmg2(u, lambda_star, gain2) != 0)
          {
            memcpy(work, tmp_work, sizeof_int * (tmp_depth + 2));
            curr_depth = tmp_depth;
            for (i = 0; i < curr_depth + 2; i++)
              active[work[i]] = true;
            // reconstruct the whole gradient
            reconstruct_gradient();
            // reset active set size
            active_size = l;
            if (select_working_set_hmg(u, lambda_star, gain_hmg) == 0)
              counter = 1;
            else
            {
              if (select_working_set_hmg2(u, lambda_star, gain2) == 0)
                counter = 1;
              else
                break;
            }
          }
        }
        else
        {
          // try to improve the working set

          // store current direction, step size and the working set
          working_set_size = curr_depth + 2;
          memcpy(tmp_work, work, sizeof_int * working_set_size);
          memcpy(tmp_u, u, sizeof_double * working_set_size);
          tmp_depth = curr_depth;
          tmp_working_set_size = working_set_size;
          tmp_lambda = lambda_star;
          for (i = 0; i < working_set_size; i++)
          {
            active[work[i]] = false;
            work[i] = -1;
          }
          curr_depth = 0;
          working_set_size = 0;
          if (select_working_set_hmg2(u, lambda_star, gain2) == 0)
          {
            if (gain_hmg > gain2)
            {
              for (i = 0; i < working_set_size; i++)
                active[work[i]] = false;
              curr_depth = tmp_depth;
              working_set_size = tmp_working_set_size;
              lambda_star = tmp_lambda;
              memcpy(work, tmp_work, sizeof_int * tmp_working_set_size);
              memcpy(u, tmp_u, sizeof_double * working_set_size);
              for (i = 0; i < working_set_size; i++)
                active[work[i]] = true;
            }
          }
          else
          {
            curr_depth = tmp_depth;
            working_set_size = tmp_working_set_size;
            lambda_star = tmp_lambda;
            memcpy(u, tmp_u, sizeof_double * working_set_size);
            memcpy(work, tmp_work, sizeof_int * tmp_working_set_size);
            for (i = 0; i < working_set_size; i++)
              active[work[i]] = true;
          }
        }
      }
      else if (select_working_set_hmg2(u, lambda_star, gain2) != 0)
      {
        reconstruct_gradient();
        active_size = l;
        if (select_working_set_hmg2(u, lambda_star, gain2) == 0)
          counter = 1;
        else
          break;
      }
    }
    iter++;

    /////////////////////////////////////////////////////////////

    // work array - the chosen working set
    // working_set_size - size of the working set

    // shift old alpha's, G's and G_bar's
    if (max_depth > 0)
    {
      n_conjugate++;
      next_depth = min(curr_depth + 1, max_depth);
      tmp_alpha = alpha_cg[next_depth];
      tmp_G = G_cg[next_depth];
      tmp_G_bar = G_bar_cg[next_depth];
      tmp_status = alpha_status_cg[next_depth];

      for (i = next_depth; i > 0; i--)
      {
        alpha_cg[i] = alpha_cg[i - 1];
        G_cg[i] = G_cg[i - 1];
        G_bar_cg[i] = G_bar_cg[i - 1];
        alpha_status_cg[i] = alpha_status_cg[i - 1];
      }

      if (!done_shrinking || (curr_depth == max_depth && curr_depth == prev_depth))
      {
        memcpy(tmp_alpha, alpha_cg[0], active_size * sizeof_double);
        memcpy(tmp_G, G_cg[0], active_size * sizeof_double);
        if ((iter - last_updated_G_bar <= max_depth) || (curr_depth > prev_depth))
          memcpy(tmp_G_bar, G_bar_cg[0], l * sizeof_double);
        memcpy(tmp_status, alpha_status_cg[0], active_size * sizeof_char);
        done_shrinking = false;
      }
      else
      {
        memcpy(tmp_alpha, alpha_cg[0], l * sizeof_double);
        memcpy(tmp_G, G_cg[0], l * sizeof_double);
        memcpy(tmp_G_bar, G_bar_cg[0], l * sizeof_double);
        memcpy(tmp_status, alpha_status_cg[0], l * sizeof_char);
      }

      alpha_cg[0] = tmp_alpha;
      G_cg[0] = tmp_G;
      G_bar_cg[0] = tmp_G_bar;
      alpha_status_cg[0] = tmp_status;
    }
    else
    {
      old_alpha[0] = alpha_cg[0][work[0]];
      old_alpha[1] = alpha_cg[0][work[1]];
      old_upper_bound[0] = is_upper_bound_cg(work[0], 0);
      old_upper_bound[1] = is_upper_bound_cg(work[1], 0);
    }

    // update alpha's and alpha_status
    for (i = 0; i < curr_depth + 2; i++)
    {
      work_i = work[i];
      alpha_cg[0][work_i] += lambda_star * u[i];
      update_alpha_status_cg(work_i, 0);
    }

    // check if we are at the corner
    corner = true;
    bumped = false;
    for (i = 0; i < curr_depth + 2; i++)
    {
      work_i = work[i];
      alpha_i = alpha_cg[0][work_i];
      C_i = get_C(work_i);
      if (alpha_i >= 1e-8 * C_i && alpha_i <= C_i - 1e-8)
      {
        corner = false;
        break;
      }
      else
        bumped = true;
    }

    // update G and G_bar
    if (max_depth > 0)
      for (i = 0; i < curr_depth + 2; i++)
      {
        work_i = work[i];
        delta_alpha_i = alpha_cg[0][work_i] - alpha_cg[1][work_i];
        Q_i = Q.get_Q(work_i, active_size);
        for (j = 0; j < active_size; j++)
          G_cg[0][j] += Q_i[j] * delta_alpha_i;
        upper_bound_i = is_upper_bound_cg(work_i, 0);
        if (upper_bound_i != is_upper_bound_cg(work_i, 1))
        {
          last_updated_G_bar = iter;
          C_i = get_C(work_i);
          Q_i = Q.get_Q(work_i, l);
          if (upper_bound_i)
            for (j = 0; j < l; j++)
              G_bar_cg[0][j] += Q_i[j] * C_i;
          else
            for (j = 0; j < l; j++)
              G_bar_cg[0][j] -= Q_i[j] * C_i;
        }
      }
    else
      for (i = 0; i < curr_depth + 2; i++)
      {
        work_i = work[i];
        delta_alpha_i = alpha_cg[0][work_i] - old_alpha[i];
        Q_i = Q.get_Q(work_i, active_size);
        for (j = 0; j < active_size; j++)
          G_cg[0][j] += Q_i[j] * delta_alpha_i;
        upper_bound_i = is_upper_bound_cg(work_i, 0);
        if (upper_bound_i != old_upper_bound[i])
        {
          last_updated_G_bar = iter;
          C_i = get_C(work_i);
          Q_i = Q.get_Q(work_i, l);
          if (upper_bound_i)
            for (j = 0; j < l; j++)
              G_bar_cg[0][j] += Q_i[j] * C_i;
          else
            for (j = 0; j < l; j++)
              G_bar_cg[0][j] -= Q_i[j] * C_i;
        }
      }
  }

  // calculate rho
  si->rho = calculate_rho_cg();

  // calculate objective value
  double v = 0;
  for (i = 0; i < l; i++)
    v += alpha_cg[0][i] * (G_cg[0][i] + p[i]);

  si->obj = v / 2;
  info("Objective value= %f\n", si->obj);

  // put back the solution
  for (i = 0; i < l; i++)
    alpha_[active_set[i]] = alpha_cg[0][i];

  si->upper_bound_p = Cp;
  si->upper_bound_n = Cn;

  info("\noptimization finished, #iter = %d\n", iter);

  delete[] p;
  delete[] y;
  delete[] u;
  delete[] tmp_u;
  delete[] work;
  delete[] tmp_work;
  delete[] active_set;
  delete[] active;
  //
  for(i=0; i<max_depth+1; i++) {
    delete[] alpha_cg[i];
    delete[] alpha_status_cg[i]; 
    delete[] G_cg[i];
    delete[] G_bar_cg[i];
  }
  delete[] alpha_cg;
  delete[] alpha_status_cg;
  delete[] G_cg;
  delete[] G_bar_cg;
  
}

void Solver::Solve(int l, const QMatrix &Q, const double *p_, const schar *y_,
                   double *alpha_, double Cp, double Cn, double eps,
                   SolutionInfo *si, int shrinking)
{
  this->l = l;
  this->Q = &Q;
  QD = Q.get_QD();
  clone(p, p_, l);
  clone(y, y_, l);
  clone(alpha, alpha_, l);
  this->Cp = Cp;
  this->Cn = Cn;
  this->eps = eps;
  unshrink = false;

  // initialize alpha_status
  {
    alpha_status = new char[l];
    for (int i = 0; i < l; i++)
      update_alpha_status(i);
  }

  // initialize active set (for shrinking)
  {
    active_set = new int[l];
    for (int i = 0; i < l; i++)
      active_set[i] = i;
    active_size = l;
  }

  // initialize gradient
  {
    G = new double[l];
    G_bar = new double[l];
    int i;
    for (i = 0; i < l; i++)
    {
      G[i] = p[i];
      G_bar[i] = 0;
    }
    for (i = 0; i < l; i++)
      if (!is_lower_bound(i))
      {
        const Qfloat *Q_i = Q.get_Q(i, l);
        double alpha_i = alpha[i];
        int j;
        for (j = 0; j < l; j++)
          G[j] += alpha_i * Q_i[j];
        if (is_upper_bound(i))
          for (j = 0; j < l; j++)
            G_bar[j] += get_C(i) * Q_i[j];
      }
  }

  // optimization step

  iter = 0;
  int counter = min(l, 1000) + 1;
  int nbumped = 0;

  while (1)
  {
    // show progress and do shrinking

    if (--counter == 0)
    {
      counter = min(l, 1000);
      if (shrinking)
        do_shrinking();
      info(".");
    }

    int i, j;
    if (select_working_set(i, j) != 0)
    {
      // reconstruct the whole gradient
      reconstruct_gradient();
      // reset active set size and check
      active_size = l;
      info("*");
      if (select_working_set(i, j) != 0)
        break;
      else
        counter = 1; // do shrinking next iteration
    }

    ++iter;

    // update alpha[i] and alpha[j], handle bounds carefully

    const Qfloat *Q_i = Q.get_Q(i, active_size);
    const Qfloat *Q_j = Q.get_Q(j, active_size);

    double C_i = get_C(i);
    double C_j = get_C(j);

    double old_alpha_i = alpha[i];
    double old_alpha_j = alpha[j];

    if (y[i] != y[j])
    {
      double quad_coef = Q_i[i] + Q_j[j] + 2 * Q_i[j];
      if (quad_coef <= 0)
        quad_coef = TAU;
      double delta = (-G[i] - G[j]) / quad_coef;
      double diff = alpha[i] - alpha[j];
      alpha[i] += delta;
      alpha[j] += delta;

      if (diff > 0)
      {
        if (alpha[j] < 0)
        {
          alpha[j] = 0;
          alpha[i] = diff;
        }
      }
      else
      {
        if (alpha[i] < 0)
        {
          alpha[i] = 0;
          alpha[j] = -diff;
        }
      }
      if (diff > C_i - C_j)
      {
        if (alpha[i] > C_i)
        {
          alpha[i] = C_i;
          alpha[j] = C_i - diff;
        }
      }
      else
      {
        if (alpha[j] > C_j)
        {
          alpha[j] = C_j;
          alpha[i] = C_j + diff;
        }
      }
    }
    else
    {
      double quad_coef = Q_i[i] + Q_j[j] - 2 * Q_i[j];
      if (quad_coef <= 0)
        quad_coef = TAU;
      double delta = (G[i] - G[j]) / quad_coef;
      double sum = alpha[i] + alpha[j];
      alpha[i] -= delta;
      alpha[j] += delta;

      if (sum > C_i)
      {
        if (alpha[i] > C_i)
        {
          alpha[i] = C_i;
          alpha[j] = sum - C_i;
        }
      }
      else
      {
        if (alpha[j] < 0)
        {
          alpha[j] = 0;
          alpha[i] = sum;
        }
      }
      if (sum > C_j)
      {
        if (alpha[j] > C_j)
        {
          alpha[j] = C_j;
          alpha[i] = sum - C_j;
        }
      }
      else
      {
        if (alpha[i] < 0)
        {
          alpha[i] = 0;
          alpha[j] = sum;
        }
      }
    }

    // update G

    double delta_alpha_i = alpha[i] - old_alpha_i;
    double delta_alpha_j = alpha[j] - old_alpha_j;

    for (int k = 0; k < active_size; k++)
    {
      G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
    }

    // update alpha_status and G_bar

    {
      bool ui = is_upper_bound(i);
      bool uj = is_upper_bound(j);
      update_alpha_status(i);
      update_alpha_status(j);
      int k;
      if (ui != is_upper_bound(i))
      {
        Q_i = Q.get_Q(i, l);
        if (ui)
          for (k = 0; k < l; k++)
            G_bar[k] -= C_i * Q_i[k];
        else
          for (k = 0; k < l; k++)
            G_bar[k] += C_i * Q_i[k];
      }

      if (uj != is_upper_bound(j))
      {
        Q_j = Q.get_Q(j, l);
        if (uj)
          for (k = 0; k < l; k++)
            G_bar[k] -= C_j * Q_j[k];
        else
          for (k = 0; k < l; k++)
            G_bar[k] += C_j * Q_j[k];
      }
    }

    double bumped_i = is_lower_bound(i) || is_upper_bound(i);
    double bumped_j = is_lower_bound(j) || is_upper_bound(j);

    if ((bumped_i && !bumped_j) || (!bumped_i && bumped_j))
      nbumped++;
  }

  // calculate rho

  si->rho = calculate_rho();

  // calculate objective value
  //{
    double v = 0;
    int i;
    for (i = 0; i < l; i++)
      v += alpha[i] * (G[i] + p[i]);

    si->obj = v / 2;
    info("objective value = %f\n", si->obj);
  //}

  // put back the solution
  {
    for (int i = 0; i < l; i++)
      alpha_[active_set[i]] = alpha[i];
  }

  // juggle everything back
  /*{
    for(int i=0;i<l;i++)
    while(active_set[i] != i)
    swap_index(i,active_set[i]);
    // or Q.swap_index(i,active_set[i]);
    }*/

  si->upper_bound_p = Cp;
  si->upper_bound_n = Cn;

  info("\noptimization finished, #iter = %d\n", iter);

  delete[] p;
  delete[] y;
  delete[] alpha;
  delete[] alpha_status;
  delete[] active_set;
  delete[] G;
  delete[] G_bar;
}

int Solver::generate_direction(double *u, int n, bool new_working_set)
{
  //   return generate_direction_general(u, n);

  switch (n)
  {
  case 3:
    generate_direction3(u, new_working_set);
    return 0;
  case 4:
    generate_direction4(u, new_working_set);
    return 0;
  default:
    return generate_direction_general(u, n, new_working_set);
  }
}

int Solver::generate_direction_general(double *u, int n, bool new_working_set)
{
  int i, j, result;
  int n_minus_one = n - 1;
  int n_minus_two = n - 2;
  int start, workj, work0;
  double *A0, *Ai, *G, *G_old;

  if (new_working_set)
    start = 0;
  else
    start = n_minus_two;

  // compute A matrix
  A0 = A[0];
  for (i = start; i < n_minus_one; i++)
    A0[i] = y[work[i + 1]];
  for (i = 1; i <= n_minus_two; i++)
  {
    Ai = A[i];
    G_old = G_cg[i - 1];
    G = G_cg[i];
    for (j = start; j < n_minus_one; j++)
    {
      workj = work[j + 1];
      Ai[j] = G_old[workj] - G[workj];
    }
  }

  // compute b vector
  if (new_working_set)
  {
    work0 = work[0];
    b[0] = -y[work0];
    for (i = 1; i <= n_minus_two; i++)
      b[i] = -G_cg[i - 1][work0] + G_cg[i][work0];
  }

  result = solve_linear_system(A, b, n_minus_one);
  if (result)
    return 1;
  u[0] = 1;
  memcpy(&u[1], b, sizeof(double) * (n_minus_one));

  return 0;
}

void Solver::generate_direction3(double *u, bool new_working_set)
{
  static double a_00, a_01, a_10, a_11;
  double a_02, a_12;
  double *G0, *G1;

  G0 = G_cg[0];
  G1 = G_cg[1];
  if (new_working_set)
  {
    int working0 = work[0];
    int working1 = work[1];
    a_00 = y[working0];
    a_01 = y[working1];
    a_10 = G0[working0] - G1[working0];
    a_11 = G0[working1] - G1[working1];
  }

  int working2 = work[2];
  a_02 = y[working2];
  a_12 = G0[working2] - G1[working2];

  u[0] = a_01 * a_12 - a_02 * a_11;
  u[1] = a_02 * a_10 - a_00 * a_12;
  u[2] = a_00 * a_11 - a_01 * a_10;
}

void Solver::generate_direction4(double *u, bool new_working_set)
{
  static double a_00, a_01, a_02, a_10, a_11, a_12, a_20, a_21, a_22, m_01, m_02, m_12;
  double *G0, *G1, *G2;

  G0 = G_cg[0];
  G1 = G_cg[1];
  G2 = G_cg[2];

  int working0 = work[0];
  int working1 = work[1];
  int working2 = work[2];
  int working3 = work[3];

  double G_10 = G1[working0];
  double G_11 = G1[working1];
  double G_12 = G1[working2];
  double G_13 = G1[working3];

  if (new_working_set)
  {
    a_00 = y[working0];
    a_01 = y[working1];
    a_02 = y[working2];
    a_10 = G0[working0] - G_10;
    a_11 = G0[working1] - G_11;
    a_12 = G0[working2] - G_12;
    a_20 = G_10 - G2[working0];
    a_21 = G_11 - G2[working1];
    a_22 = G_12 - G2[working2];

    m_01 = a_10 * a_21 - a_11 * a_20;
    m_02 = a_10 * a_22 - a_12 * a_20;
    m_12 = a_11 * a_22 - a_12 * a_21;
  }

  double a_03 = y[working3];
  double a_13 = G0[working3] - G_13;
  double a_23 = G_13 - G2[working3];

  double m_03 = a_10 * a_23 - a_13 * a_20;
  double m_13 = a_11 * a_23 - a_13 * a_21;
  double m_23 = a_12 * a_23 - a_13 * a_22;

  u[0] = a_01 * m_23 - a_02 * m_13 + a_03 * m_12;
  u[1] = -a_00 * m_23 + a_02 * m_03 - a_03 * m_02;
  u[2] = a_00 * m_13 - a_01 * m_03 + a_03 * m_01;
  u[3] = -a_00 * m_12 + a_01 * m_02 - a_02 * m_01;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
  // return i,j such that
  // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  // j: minimizes the decrease of obj value
  //    (if quadratic coefficeint <= 0, replace it with tau)
  //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

  double Gmax = -INF;
  double Gmax2 = -INF;
  int Gmax_idx = -1;
  int Gmin_idx = -1;
  double obj_diff_min = INF;

  for (int t = 0; t < active_size; t++)
    if (y[t] == +1)
    {
      if (!is_upper_bound(t))
        if (-G[t] >= Gmax)
        {
          Gmax = -G[t];
          Gmax_idx = t;
        }
    }
    else
    {
      if (!is_lower_bound(t))
        if (G[t] >= Gmax)
        {
          Gmax = G[t];
          Gmax_idx = t;
        }
    }

  int i = Gmax_idx;
  const Qfloat *Q_i = NULL;
  if (i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
    Q_i = Q->get_Q(i, active_size);

  for (int j = 0; j < active_size; j++)
  {
    if (y[j] == +1)
    {
      if (!is_lower_bound(j))
      {
        double grad_diff = Gmax + G[j];
        if (G[j] >= Gmax2)
          Gmax2 = G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = Q_i[i] + QD[j] - 2.0 * y[i] * Q_i[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff * grad_diff) / quad_coef;
          else
            obj_diff = -(grad_diff * grad_diff) / TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
    else
    {
      if (!is_upper_bound(j))
      {
        double grad_diff = Gmax - G[j];
        if (-G[j] >= Gmax2)
          Gmax2 = -G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = Q_i[i] + QD[j] + 2.0 * y[i] * Q_i[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff * grad_diff) / quad_coef;
          else
            obj_diff = -(grad_diff * grad_diff) / TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if (Gmax + Gmax2 < eps)
    return 1;

  out_i = Gmax_idx;
  out_j = Gmin_idx;

  return 0;
}

int Solver::select_working_set_hmg2(double *u, double &lambda_star, double &best_gain)
{
  for (int i = 0; i < curr_depth + 2; i++)
  {
    active[work[i]] = false;
    work[i] = -1;
  }
  curr_depth = -1;

  if (select_working_set_first_order())
    return 1;

  // generic situation: use the MG selection
  int a, b, bb, out_i = -1, out_j = -1;
  double da, db; // diagonal entries of Q
  double ga, gb; // gradient in coordinates a and b
  double gt, gs; // gradient in coordinate a+b or a-b
  double alpha_a, alpha_b;
  double progress;
  double nu;
  double Ca, Cb;
  double lambda;
  Qfloat *q;

  double best = 0.0;
  double g_best = 0.0;

  // try combinations with the last working set
  for (bb = 0; bb < 2; bb++)
  {
    b = work[bb];

    q = Q->get_Q(b, active_size);

    db = QD[b];
    Cb = get_C(b);
    gb = G_cg[0][b];
    alpha_b = alpha_cg[0][b];

    for (a = 0; a < active_size; a++)
    {
      if (a == b)
        continue;
      da = QD[a];
      Ca = get_C(a);
      ga = G_cg[0][a];
      alpha_a = alpha_cg[0][a];
      nu = q[a];

      if (y[a] * y[b] > 0.0)
      {
        lambda = da + db - 2.0 * nu;
        gs = gt = (ga - gb) / lambda;
        if (gs > 0.0)
        {
          if (alpha_a <= 1e-12 || alpha_b >= Cb - 1e-12)
            continue;
          if (gs < -alpha_a)
            gs = -alpha_a;
          if (gs < alpha_b - Cb)
            gs = alpha_b - Cb;
        }
        else
        {
          if (alpha_a >= Ca - 1e-12 || alpha_b <= 1e-12)
            continue;
          if (gs > Ca - alpha_a)
            gs = Ca - alpha_a;
          if (gs > alpha_b)
            gs = alpha_b;
        }
        progress = gs * (2.0 * gt - gs) * lambda;
      }
      else
      {
        lambda = da + db + 2.0 * nu;
        gs = gt = (ga + gb) / lambda;
        if (gs > 0.0)
        {
          if (alpha_a <= 1e-12 || alpha_b <= 1e-12)
            continue;
          if (gs < -alpha_a)
            gs = -alpha_a;
          if (gs < -alpha_b)
            gs = -alpha_b;
        }
        else
        {
          if (alpha_a >= Ca - 1e-12 || alpha_b >= Cb - 1e-12)
            continue;
          if (gs > Ca - alpha_a)
            gs = Ca - alpha_a;
          if (gs > Cb - alpha_b)
            gs = Cb - alpha_b;
        }
        progress = gs * (2.0 * gt - gs) * lambda;
      }

      // select the largest progress
      if (progress > best)
      {
        best = progress;
        g_best = gs;
        out_i = a;
        out_j = b;
      }
    }
  }

  // stopping condition
  if (fabs(g_best) < eps)
    return 1; // optimal

  best_gain = best;
  work[0] = out_i;
  work[1] = out_j;
  active[out_i] = true;
  active[out_j] = true;

  compute_step_first_order(u, lambda_star);
  curr_depth = 0;
  return 0;
}

bool Solver::feasible_direction(double *u)
{
  int i;
  double alpha_i;
  double *alpha0 = alpha_cg[0];

  for (i = 0; i < curr_depth + 3; i++)
  {
    alpha_i = alpha0[work[i]];
    if ((u[i] > 0 && alpha_i >= get_C(work[i])) || (u[i] < 0 && alpha_i <= 0))
      return false;
  }

  return true;
}

// select k given i and j
int Solver::select_working_set_incrementally(double *best_u, double &out_lambda_star, double &final_gain)
{
  //printf("Solver select_working_set_incrementally \n");
  int i, j, k, best_indx = -1;
  double gain, best_gain = 0;
  double nominator, denominator;
  //double A[curr_depth + 3], B[curr_depth + 3], lambda_init, lambda_star;
  //printf("curr_depth + 3 %d \n", curr_depth + 3);
  // double A[10], B[10], lambda_init, lambda_star;
  double A[3], B[3], lambda_init, lambda_star;
  bool feasible_u, feasible_minus_u, negligible;
  //double u[curr_depth + 3], minus_u[curr_depth + 3], *curr_u;
  double u[10], minus_u[10], *curr_u;
  //Qfloat *Q_row[curr_depth + 3];
  Qfloat *Q_row[10];
  double *G0 = G_cg[0];
  Qfloat *Q_row_i;
  double *alpha0;

  for (i = 0; i < curr_depth + 2; i++)
    Q_row[i] = Q->get_Q(work[i], active_size);

  for (k = 0; k < active_size; k++)
  {

    // check that k is not in the working set
    if (active[k])
      continue;

    // generate direction
    work[curr_depth + 2] = k;
    if (generate_direction(u, curr_depth + 3, k == 0))
      continue;

    // check if u or -u is feasible
    for (i = 0; i < curr_depth + 3; i++)
      minus_u[i] = -u[i];

    feasible_u = feasible_direction(u);
    feasible_minus_u = feasible_direction(minus_u);

    if (!feasible_u && !feasible_minus_u)
      continue;

    // choose which of {u,minus_u} is a descent direction
    nominator = 0;
    for (i = 0; i < curr_depth + 3; i++)
      nominator += G0[work[i]] * u[i];

    if (nominator < 0)
      if (feasible_u)
        curr_u = u;
      else
        continue;
    else if (feasible_minus_u)
    {
      nominator *= -1;
      curr_u = minus_u;
    }
    else
      continue;

    // compute denominator
    denominator = 0;
    for (i = 0; i < curr_depth + 3; i++)
    {
      denominator += curr_u[i] * curr_u[i] * QD[work[i]];
      Q_row_i = Q_row[i];
      for (j = i + 1; j < curr_depth + 3; j++)
        denominator += 2 * curr_u[i] * curr_u[j] * Q_row_i[work[j]];
    }

    if (denominator <= 0)
      denominator = TAU;

    lambda_init = -nominator / denominator;
    lambda_star = lambda_init;

    for (i = 0; i < curr_depth + 3; i++)
      if (curr_u[i] > 0)
      {
        A[i] = 0;
        B[i] = get_C(work[i]);
      }
      else
      {
        A[i] = get_C(work[i]);
        B[i] = 0;
      }

    alpha0 = alpha_cg[0];
    for (i = 0; i < curr_depth + 3; i++)
      lambda_star = max(lambda_star, (A[i] - alpha0[work[i]]) / curr_u[i]);
    for (i = 0; i < curr_depth + 3; i++)
      lambda_star = min(lambda_star, (B[i] - alpha0[work[i]]) / curr_u[i]);

    if (fabs(lambda_star) < 1e-3)
      continue;
    negligible = false;
    for (i = 0; i < curr_depth + 3; i++)
      if (fabs(lambda_star * curr_u[i]) < 1e-3)
      {
        negligible = true;
        break;
      }
    if (negligible)
      continue;

    gain = denominator * lambda_star * (2 * lambda_init - lambda_star);

    if (gain > best_gain)
    {
      best_gain = gain;
      best_indx = k;
      memcpy(best_u, curr_u, sizeof_double * (curr_depth + 3));
      out_lambda_star = lambda_star;
    }
  }

  final_gain = best_gain;

  if (best_indx != -1)
  {
    work[curr_depth + 2] = best_indx;
    active[best_indx] = true;
    curr_depth++;
    return 0;
  }
  else
  {
    work[curr_depth + 2] = -1;
    return 1;
  }
}

// select working set incrementally
int Solver::select_working_set_hmg(double *u, double &lambda_star, double &best_gain)
{
  int i;

  if (curr_depth != max_depth)
    return select_working_set_incrementally(u, lambda_star, best_gain);

  // throw away the oldest active example and select the working set incrementally
  active[work[0]] = false;
  for (i = 0; i <= curr_depth; i++)
    work[i] = work[i + 1];
  work[curr_depth + 1] = -1;
  curr_depth--;

  return select_working_set_incrementally(u, lambda_star, best_gain);
}

int Solver::wss_first_order(double *u, double &lambda_star)
{
  for (int i = 0; i < curr_depth + 2; i++)
  {
    active[work[i]] = false;
    work[i] = -1;
  }
  curr_depth = -1;
  if (select_working_set_first_order())
    return 1;
  active[work[0]] = true;
  active[work[1]] = true;
  compute_step_first_order(u, lambda_star);
  curr_depth = 0;
  return 0;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set_first_order()
{

  // return i,j which maximize -grad(f)^T d , under constraint
  // if alpha_i == C, d != +1
  // if alpha_i == 0, d != -1

  double Gmax1 = -INF; // max { -grad(f)_i * d | y_i*d = +1 }
  int Gmax1_idx = -1;

  double Gmax2 = -INF; // max { -grad(f)_i * d | y_i*d = -1 }//
  int Gmax2_idx = -1;
  double *G0 = G_cg[0];

  for (int i = 0; i < active_size; i++)
  {
    if (y[i] == +1)
    { // y = +1
      if (!is_upper_bound_cg(i, 0))
      { // d = +1
        if (-G0[i] >= Gmax1)
        {
          Gmax1 = -G0[i];
          Gmax1_idx = i;
        }
      }
      if (!is_lower_bound_cg(i, 0))
      { // d = -1
        if (G0[i] >= Gmax2)
        {
          Gmax2 = G0[i];
          Gmax2_idx = i;
        }
      }
    }
    else
    { // y = -1
      if (!is_upper_bound_cg(i, 0))
      { // d = +1
        if (-G0[i] >= Gmax2)
        {
          Gmax2 = -G0[i];
          Gmax2_idx = i;
        }
      }
      if (!is_lower_bound_cg(i, 0))
      { // d = -1
        if (G0[i] >= Gmax1)
        {
          Gmax1 = G0[i];
          Gmax1_idx = i;
        }
      }
    }
  }

  if (Gmax1 + Gmax2 < eps)
    return 1;

  work[0] = Gmax1_idx;
  work[1] = Gmax2_idx;

  return 0;
}

void Solver::compute_step_first_order(double *u, double &lambda_star)
{
  int out_i = work[0];
  int out_j = work[1];

  Qfloat *Q_i = Q->get_Q(out_i, active_size);
  Qfloat *Q_j = Q->get_Q(out_j, active_size);

  double C_i = get_C(out_i);
  double C_j = get_C(out_j);
  double old_alpha_i = alpha_cg[0][out_i];
  double old_alpha_j = alpha_cg[0][out_j];
  double new_alpha_i, new_alpha_j;
  if (y[out_i] != y[out_j])
  {
    double quad_coef = Q_i[out_i] + Q_j[out_j] + 2 * Q_i[out_j];
    if (quad_coef <= 0)
      quad_coef = TAU;
    lambda_star = (-G_cg[0][out_i] - G_cg[0][out_j]) / quad_coef;
    u[0] = 1;
    u[1] = 1;
    double diff = old_alpha_i - old_alpha_j;
    new_alpha_i = old_alpha_i + lambda_star;
    new_alpha_j = old_alpha_j + lambda_star;
    if (diff > 0)
    {
      if (new_alpha_j < 0)
        lambda_star = -old_alpha_j;
    }
    else
    {
      if (new_alpha_i < 0)
        lambda_star = -old_alpha_i;
    }
    if (diff > C_i - C_j)
    {
      if (new_alpha_i > C_i)
        lambda_star = C_i - old_alpha_i;
    }
    else
    {
      if (new_alpha_j > C_j)
        lambda_star = C_j - old_alpha_j;
    }
  }
  else
  {
    double quad_coef = Q_i[out_i] + Q_j[out_j] - 2 * Q_i[out_j];
    if (quad_coef <= 0)
      quad_coef = TAU;
    lambda_star = (G_cg[0][out_i] - G_cg[0][out_j]) / quad_coef;
    u[0] = -1;
    u[1] = 1;
    double sum = old_alpha_i + old_alpha_j;
    new_alpha_i = old_alpha_i - lambda_star;
    new_alpha_j = old_alpha_j + lambda_star;

    if (sum > C_i)
    {
      if (new_alpha_i > C_i)
        lambda_star = -C_i + old_alpha_i;
    }
    else
    {
      if (new_alpha_j < 0)
        lambda_star = -old_alpha_j;
    }
    if (sum > C_j)
    {
      if (new_alpha_j > C_j)
        lambda_star = C_j - old_alpha_j;
    }
    else
    {
      if (new_alpha_i < 0)
        lambda_star = old_alpha_i;
    }
  }
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
  if (is_upper_bound(i))
  {
    if (y[i] == +1)
      return (-G[i] > Gmax1);
    else
      return (-G[i] > Gmax2);
  }
  else if (is_lower_bound(i))
  {
    if (y[i] == +1)
      return (G[i] > Gmax2);
    else
      return (G[i] > Gmax1);
  }
  else
    return (false);
}

bool Solver::be_shrunk_cg(int i, double Gmax1, double Gmax2)
{
  if (is_upper_bound_cg(i, 0))
  {
    if (y[i] == +1)
      return (-G_cg[0][i] > Gmax1);
    else
      return (-G_cg[0][i] > Gmax2);
  }
  else if (is_lower_bound_cg(i, 0))
  {
    if (y[i] == +1)
      return (G_cg[0][i] > Gmax2);
    else
      return (G_cg[0][i] > Gmax1);
  }
  else
    return (false);
}

void Solver::do_shrinking()
{
  int i;
  double Gmax1 = -INF; // max { -y_i * grad(f)_i | i in I_up(\alpha) }
  double Gmax2 = -INF; // max { y_i * grad(f)_i | i in I_low(\alpha) }

  // find maximal violating pair first
  for (i = 0; i < active_size; i++)
  {
    if (y[i] == +1)
    {
      if (!is_upper_bound(i))
      {
        if (-G[i] >= Gmax1)
          Gmax1 = -G[i];
      }
      if (!is_lower_bound(i))
      {
        if (G[i] >= Gmax2)
          Gmax2 = G[i];
      }
    }
    else
    {
      if (!is_upper_bound(i))
      {
        if (-G[i] >= Gmax2)
          Gmax2 = -G[i];
      }
      if (!is_lower_bound(i))
      {
        if (G[i] >= Gmax1)
          Gmax1 = G[i];
      }
    }
  }

  if (unshrink == false && Gmax1 + Gmax2 <= eps * 10)
  {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
    info("*");
  }

  for (i = 0; i < active_size; i++)
  {
    if (be_shrunk(i, Gmax1, Gmax2))
    {
      active_size--;
      while (active_size > i)
      {
        if (!be_shrunk(active_size, Gmax1, Gmax2))
        {
          swap_index(i, active_size);
          break;
        }
        active_size--;
      }
    }
  }
}

bool Solver::do_shrinking_cg()
{
  int i, j;
  double Gmax1 = -INF; // max { -y_i * grad(f)_i | i in I_up(\alpha) }
  double Gmax2 = -INF; // max { y_i * grad(f)_i | i in I_low(\alpha) }
  bool done_shrinking = false;

  // find maximal violating pair first
  for (i = 0; i < active_size; i++)
    if (y[i] == 1)
    {
      if (!is_upper_bound_cg(i, 0) && -G_cg[0][i] >= Gmax1)
        Gmax1 = -G_cg[0][i];
      if (!is_lower_bound_cg(i, 0) && G_cg[0][i] >= Gmax2)
        Gmax2 = G_cg[0][i];
    }
    else
    {
      if (!is_upper_bound_cg(i, 0) && -G_cg[0][i] >= Gmax2)
        Gmax2 = -G_cg[0][i];
      if (!is_lower_bound_cg(i, 0) && G_cg[0][i] >= Gmax1)
        Gmax1 = G_cg[0][i];
    }

  if (unshrink == false && Gmax1 + Gmax2 <= eps * 10)
  {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
    info("*");
  }

  for (i = 0; i < active_size; i++)
  {
    if (be_shrunk_cg(i, Gmax1, Gmax2))
    {
      active_size--;
      while (active_size > i)
      {
        if (!be_shrunk_cg(active_size, Gmax1, Gmax2))
        {
          swap_index(i, active_size);
          done_shrinking = true;
          break;
        }
        else
        {
          if (active[active_size])
          {
            active[active_size] = false;
            for (j = 0; j < curr_depth + 2; j++)
              if (work[j] == active_size)
              {
                work[j] = -1;
                break;
              }
          }
          active_size--;
        }
      }
    }
  }
  // shrink working set
  if (done_shrinking)
  {
    for (i = 0; i < curr_depth + 2; i++)
    {
      active[work[i]] = false;
      work[i] = -1;
    }
    curr_depth = -1;
  }

  return done_shrinking;
}

double Solver::calculate_rho()
{
  double r;
  int nr_free = 0;
  double ub = INF, lb = -INF, sum_free = 0;
  for (int i = 0; i < active_size; i++)
  {
    double yG = y[i] * G[i];

    if (is_upper_bound(i))
    {
      if (y[i] == -1)
        ub = min(ub, yG);
      else
        lb = max(lb, yG);
    }
    else if (is_lower_bound(i))
    {
      if (y[i] == +1)
        ub = min(ub, yG);
      else
        lb = max(lb, yG);
    }
    else
    {
      ++nr_free;
      sum_free += yG;
    }
  }

  if (nr_free > 0)
    r = sum_free / nr_free;
  else
    r = (ub + lb) / 2;

  return r;
}

double Solver::calculate_rho_cg()
{
  double r;
  int nr_free = 0;
  double ub = INF, lb = -INF, sum_free = 0;
  for (int i = 0; i < active_size; i++)
  {
    double yG = y[i] * G_cg[0][i];

    if (is_upper_bound_cg(i, 0))
    {
      if (y[i] == -1)
        ub = min(ub, yG);
      else
        lb = max(lb, yG);
    }
    else if (is_lower_bound_cg(i, 0))
    {
      if (y[i] == +1)
        ub = min(ub, yG);
      else
        lb = max(lb, yG);
    }
    else
    {
      ++nr_free;
      sum_free += yG;
    }
  }

  if (nr_free > 0)
    r = sum_free / nr_free;
  else
    r = (ub + lb) / 2;

  return r;
}
// Solver_NU.h
//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
  Solver_NU() {}
  void Solve(int l, const QMatrix &Q, const double *p, const schar *y,
             double *alpha, double Cp, double Cn, double eps,
             Solver::SolutionInfo *si, int shrinking)
  {
    this->si = si;
    Solver::Solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking);
  }

private:
  Solver::SolutionInfo *si;
  int select_working_set(int &i, int &j);
  double calculate_rho();
  bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
  void do_shrinking();
};

// Solver_NU.cpp
// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
  // return i,j such that y_i = y_j and
  // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  // j: minimizes the decrease of obj value
  //    (if quadratic coefficeint <= 0, replace it with tau)
  //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

  double Gmaxp = -INF;
  double Gmaxp2 = -INF;
  int Gmaxp_idx = -1;

  double Gmaxn = -INF;
  double Gmaxn2 = -INF;
  int Gmaxn_idx = -1;

  int Gmin_idx = -1;
  double obj_diff_min = INF;

  for (int t = 0; t < active_size; t++)
    if (y[t] == +1)
    {
      if (!is_upper_bound(t))
        if (-G[t] >= Gmaxp)
        {
          Gmaxp = -G[t];
          Gmaxp_idx = t;
        }
    }
    else
    {
      if (!is_lower_bound(t))
        if (G[t] >= Gmaxn)
        {
          Gmaxn = G[t];
          Gmaxn_idx = t;
        }
    }

  int ip = Gmaxp_idx;
  int in = Gmaxn_idx;
  const Qfloat *Q_ip = NULL;
  const Qfloat *Q_in = NULL;
  if (ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
    Q_ip = Q->get_Q(ip, active_size);
  if (in != -1)
    Q_in = Q->get_Q(in, active_size);

  for (int j = 0; j < active_size; j++)
  {
    if (y[j] == +1)
    {
      if (!is_lower_bound(j))
      {
        double grad_diff = Gmaxp + G[j];
        if (G[j] >= Gmaxp2)
          Gmaxp2 = G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = Q_ip[ip] + QD[j] - 2 * Q_ip[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff * grad_diff) / quad_coef;
          else
            obj_diff = -(grad_diff * grad_diff) / TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
    else
    {
      if (!is_upper_bound(j))
      {
        double grad_diff = Gmaxn - G[j];
        if (-G[j] >= Gmaxn2)
          Gmaxn2 = -G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = Q_in[in] + QD[j] - 2 * Q_in[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff * grad_diff) / quad_coef;
          else
            obj_diff = -(grad_diff * grad_diff) / TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx = j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if (max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps)
    return 1;

  if (y[Gmin_idx] == +1)
    out_i = Gmaxp_idx;
  else
    out_i = Gmaxn_idx;
  out_j = Gmin_idx;

  return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
  if (is_upper_bound(i))
  {
    if (y[i] == +1)
      return (-G[i] > Gmax1);
    else
      return (-G[i] > Gmax4);
  }
  else if (is_lower_bound(i))
  {
    if (y[i] == +1)
      return (G[i] > Gmax2);
    else
      return (G[i] > Gmax3);
  }
  else
    return (false);
}

void Solver_NU::do_shrinking()
{
  double Gmax1 = -INF; // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
  double Gmax2 = -INF; // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
  double Gmax3 = -INF; // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
  double Gmax4 = -INF; // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

  // find maximal violating pair first
  int i;
  for (i = 0; i < active_size; i++)
  {
    if (!is_upper_bound(i))
    {
      if (y[i] == +1)
      {
        if (-G[i] > Gmax1)
          Gmax1 = -G[i];
      }
      else if (-G[i] > Gmax4)
        Gmax4 = -G[i];
    }
    if (!is_lower_bound(i))
    {
      if (y[i] == +1)
      {
        if (G[i] > Gmax2)
          Gmax2 = G[i];
      }
      else if (G[i] > Gmax3)
        Gmax3 = G[i];
    }
  }

  if (unshrink == false && max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= eps * 10)
  {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
  }

  for (i = 0; i < active_size; i++)
    if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
    {
      active_size--;
      while (active_size > i)
      {
        if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
        {
          swap_index(i, active_size);
          break;
        }
        active_size--;
      }
    }
}

double Solver_NU::calculate_rho()
{
  int nr_free1 = 0, nr_free2 = 0;
  double ub1 = INF, ub2 = INF;
  double lb1 = -INF, lb2 = -INF;
  double sum_free1 = 0, sum_free2 = 0;

  for (int i = 0; i < active_size; i++)
  {
    if (y[i] == +1)
    {
      if (is_upper_bound(i))
        lb1 = max(lb1, G[i]);
      else if (is_lower_bound(i))
        ub1 = min(ub1, G[i]);
      else
      {
        ++nr_free1;
        sum_free1 += G[i];
      }
    }
    else
    {
      if (is_upper_bound(i))
        lb2 = max(lb2, G[i]);
      else if (is_lower_bound(i))
        ub2 = min(ub2, G[i]);
      else
      {
        ++nr_free2;
        sum_free2 += G[i];
      }
    }
  }

  double r1, r2;
  if (nr_free1 > 0)
    r1 = sum_free1 / nr_free1;
  else
    r1 = (ub1 + lb1) / 2;

  if (nr_free2 > 0)
    r2 = sum_free2 / nr_free2;
  else
    r2 = (ub2 + lb2) / 2;

  si->r = (r1 + r2) / 2;
  return (r1 - r2) / 2;
}

// Solver_plus.h
// An SMO and conjugate SMO algorithm for SVM+
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver_plus
{
public:
  Solver_plus(int optimizer_)
  {
    if (optimizer_ == -1)
      conjugate = false;
    else
    {
      conjugate = true;
      max_depth = optimizer_;

      A = new double *[max_depth + 5];
      for (int i = 0; i <= max_depth + 4; i++)
        A[i] = new double[max_depth + 5];
      b = new double[max_depth + 5];
    }

    sizeof_double = sizeof(double);
    sizeof_char = sizeof(char);
    sizeof_int = sizeof(int);
  };
  virtual ~Solver_plus()
  {
    if (conjugate)
    {
      for (int i = 0; i <= max_depth + 4; i++)
        delete[] A[i];
      delete[] A;
      delete[] b;
    }
  };

  void Solve_plus(int l, const QMatrix &Q, const QMatrix &Q_star, const QMatrix &Q_star_beta, const schar *y_,
                  double *alpha_, double *beta_, double Cp, double Cn, double tau, double eps,
                  Solver::SolutionInfo *si, int shrinking);
  void Solve_plus_cg(int l, const QMatrix &Q, const QMatrix &Q_star, const QMatrix &Q_star_beta, const schar *y_,
                     double *alpha_, double *beta_, double Cp, double Cn, double tau, double eps,
                     Solver::SolutionInfo *si, int shrinking);

protected:
  int active_size;
  int active_size_beta;
  schar *y;
  double *G; // gradient of objective function
  double **G_cg;
  double *g;
  double **g_cg;
  double *g_beta;
  double **g_beta_cg;
  double *g_init;
  double *g_beta_init;
  int *work;
  enum
  {
    LOWER_BOUND,
    UPPER_BOUND,
    FREE
  };
  char *alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE
  char *beta_status;  // LOWER_BOUND, FREE
  char **alpha_status_cg;
  char **beta_status_cg;
  double *alpha;
  double *beta;
  double **alpha_cg;
  double **beta_cg;
  const QMatrix *Q;
  const QMatrix *Q_star;
  const QMatrix *Q_star_beta;
  const Qfloat *QD;
  const Qfloat *QD_star;
  const Qfloat *QD_star_beta;
  double eps;
  double Cp, Cn;
  double tau;
  double *p;
  int *active_set;
  int *active_set_beta;
  int *true_act_set;
  int *true_act_set_beta;
  double *G_bar; // gradient, if we treat free variables as 0
  int l;
  bool unshrink; // XXX

  double get_C(int i)
  {
    return (y[i] > 0) ? Cp : Cn;
  }
  void update_alpha_status(int i)
  {
    if (alpha[i] <= 1e-8)
      alpha_status[i] = LOWER_BOUND;
    else
      alpha_status[i] = FREE;
  }
  void update_beta_status(int i)
  {
    if (beta[i] <= 1e-8)
      beta_status[i] = LOWER_BOUND;
    else
      beta_status[i] = FREE;
  }
  void update_alpha_status_cg(int i, int depth)
  {
    if (alpha_cg[depth][i] <= 1e-8)
      alpha_status_cg[depth][i] = LOWER_BOUND;
    else
      alpha_status_cg[depth][i] = FREE;
  }
  void update_beta_status_cg(int i, int depth)
  {
    if (beta_cg[depth][i] <= 1e-8)
      beta_status_cg[depth][i] = LOWER_BOUND;
    else
      beta_status_cg[depth][i] = FREE;
  }
  bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
  bool is_lower_bound_beta(int i) { return beta_status[i] == LOWER_BOUND; }
  bool is_free(int i) { return alpha_status[i] == FREE; }
  bool is_free_beta(int i) { return beta_status[i] == FREE; }
  bool is_lower_bound_cg(int i, int depth) { return alpha_status_cg[depth][i] == LOWER_BOUND; }
  bool is_lower_bound_beta_cg(int i, int depth) { return beta_status_cg[depth][i] == LOWER_BOUND; }
  bool is_free_cg(int i, int depth) { return alpha_status_cg[depth][i] == FREE; }
  bool is_free_beta_cg(int i, int depth) { return beta_status_cg[depth][i] == FREE; }
  void swap_index_alpha(int i, int j);
  void swap_index_beta(int i, int j);
  void swap_index_alpha_cg(int i, int j);
  void swap_index_beta_cg(int i, int j);
  void reconstruct_gradient_plus();
  virtual int select_working_set_plus(int &set_type, int &i, int &j, int &k, int iter);
  virtual void calculate_rho_plus(double &rho, double &rho_star);
  virtual void calculate_rho_plus_cg(double &rho, double &rho_star);
  virtual void do_shrinking_plus();
  virtual bool do_shrinking_plus_cg();
  virtual int select_working_set(double *u, double &lambda_star);

private:
  bool be_shrunk_alpha(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4);
  bool be_shrunk_beta(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4);
  bool be_shrunk_alpha_cg(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4);
  bool be_shrunk_beta_cg(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4);
  void generate_direction3(double *u, bool new_working_set);
  void generate_direction4(double *u, bool new_working_set);
  int generate_direction_general(double *u, int n, bool new_working_set);
  void generate_direction4y(double *u, bool new_working_set);
  void generate_direction5y(double *u, bool new_working_set);
  int generate_direction_y_general(double *u, int n, bool new_working_set);
  int generate_direction(double *u, int n, bool new_working_set);
  int generate_direction_y(double *u, int n, bool new_working_set);
  void reconstruct_gradient_plus_cg();
  int wss_first_order(double *best_u, double &lambda_star);
  int select_working_set_plus_hmg(double *best_u, double &lambda_star, double &gain);
  int select_working_set_plus_hmg2(double *best_u, double &lambda_star, double &gain);
  int select_working_set_plus_incrementally(double *best_u, double &lambda_star, double &gain);
  bool compute_gain(int new_working_set_size, double &gain, double *curr_u, double &lambda, bool new_working_set);
  bool feasible_direction(double *u);
  double **A;
  double *b;
  int sizeof_double;
  int sizeof_int;
  int sizeof_char;
  bool conjugate;
  int max_depth;
  int curr_depth;
  int prev_depth;
  int working_set_size;
  bool *active;
  char working_set_type;
  enum
  {
    BETAS = 0,
    ALPHAS = 1,
    ALPHAS_BETAS = 2,
    ALPHAS_DIFF_SIGN = 3
  };
};

// Solver_plus.cpp
static int *tmp_work;
static double *tmp_u;

int Solver_plus::generate_direction(double *u, int n, bool new_working_set)
{
  switch (n)
  {
  case 3:
    generate_direction3(u, new_working_set);
    return 0;
  case 4:
    generate_direction4(u, new_working_set);
    return 0;
  default:
    return generate_direction_general(u, n, new_working_set);
  }
}

int Solver_plus::generate_direction_y(double *u, int n, bool new_working_set)
{
  switch (n)
  {
  case 4:
    generate_direction4y(u, new_working_set);
    return 0;
  case 5:
    generate_direction5y(u, new_working_set);
    return 0;
  default:
    return generate_direction_y_general(u, n, new_working_set);
  }
}

void Solver_plus::generate_direction3(double *u, bool new_working_set)
{
  int working2 = work[2];
  int working_new;

  double *G0 = G_cg[0];
  double *G1 = G_cg[1];
  double *g0 = g_cg[0];
  double *g1 = g_cg[1];
  double *g0beta = g_beta_cg[0];
  double *g1beta = g_beta_cg[1];
  int y_i;

  static double a0, a1;
  double a2;

  if (new_working_set)
  {
    int working0 = work[0];
    int working1 = work[1];

    if (working0 < l)
    {
      y_i = y[working0];
      a0 = y_i * G0[working0] + g0[working0] / tau - y_i * G1[working0] - g1[working0] / tau;
    }
    else
    {
      working_new = working0 - l;
      a0 = g0beta[working_new] / tau - g1beta[working_new] / tau;
    }

    if (working1 < l)
    {
      y_i = y[working1];
      a1 = y_i * G0[working1] + g0[working1] / tau - y_i * G1[working1] - g1[working1] / tau;
    }
    else
    {
      working_new = working1 - l;
      a1 = g0beta[working_new] / tau - g1beta[working_new] / tau;
    }
  }

  if (working2 < l)
  {
    y_i = y[working2];
    a2 = y_i * G0[working2] + g0[working2] / tau - y_i * G1[working2] - g1[working2] / tau;
  }
  else
  {
    working_new = working2 - l;
    a2 = g0beta[working_new] / tau - g1beta[working_new] / tau;
  }

  u[0] = a2 - a1;
  u[1] = a0 - a2;
  u[2] = a1 - a0;
}

void Solver_plus::generate_direction4(double *u, bool new_working_set)
{
  int working3 = work[3];
  int working_new;
  int y_i;

  double *G0 = G_cg[0];
  double *G1 = G_cg[1];
  double *G2 = G_cg[2];

  double *g0 = g_cg[0];
  double *g1 = g_cg[1];
  double *g2 = g_cg[2];

  double *g0beta = g_beta_cg[0];
  double *g1beta = g_beta_cg[1];
  double *g2beta = g_beta_cg[2];

  double G0old;
  double a14, a24;
  static double a11, a12, a13, a21, a22, a23;
  static double d1, d2, d4;
  double d3, d5, d6;

  if (new_working_set)
  {
    int working0 = work[0];
    int working1 = work[1];
    int working2 = work[2];
    if (working0 < l)
    {
      y_i = y[working0];
      G0old = y_i * G1[working0] + g1[working0] / tau;
      a11 = y_i * G0[working0] + g0[working0] / tau - G0old;
      a21 = G0old - y_i * G2[working0] - g2[working0] / tau;
    }
    else
    {
      working_new = working0 - l;
      G0old = g1beta[working_new] / tau;
      a11 = g0beta[working_new] / tau - G0old;
      a21 = G0old - g2beta[working_new] / tau;
    }

    if (working1 < l)
    {
      y_i = y[working1];
      G0old = y_i * G1[working1] + g1[working1] / tau;
      a12 = y_i * G0[working1] + g0[working1] / tau - G0old;
      a22 = G0old - y_i * G2[working1] - g2[working1] / tau;
    }
    else
    {
      working_new = working1 - l;
      G0old = g1beta[working_new] / tau;
      a12 = g0beta[working_new] / tau - G0old;
      a22 = G0old - g2beta[working_new] / tau;
    }

    if (working2 < l)
    {
      y_i = y[working2];
      G0old = y_i * G1[working2] + g1[working2] / tau;
      a13 = y_i * G0[working2] + g0[working2] / tau - G0old;
      a23 = G0old - y_i * G2[working2] - g2[working2] / tau;
    }
    else
    {
      working_new = working2 - l;
      G0old = g1beta[working_new] / tau;
      a13 = g0beta[working_new] / tau - G0old;
      a23 = G0old - g2beta[working_new] / tau;
    }

    d1 = a11 * a22 - a12 * a21;
    d2 = a11 * a23 - a13 * a21;
    d4 = a12 * a23 - a13 * a22;
  }

  if (working3 < l)
  {
    G0old = y[working3] * G1[working3] + g1[working3] / tau;
    a14 = y[working3] * G0[working3] + g0[working3] / tau - G0old;
    a24 = G0old - y[working3] * G2[working3] - g2[working3] / tau;
  }
  else
  {
    working_new = working3 - l;
    G0old = g1beta[working_new] / tau;
    a14 = g0beta[working_new] / tau - G0old;
    a24 = G0old - g2beta[working_new] / tau;
  }

  d3 = a11 * a24 - a21 * a14;
  d5 = a12 * a24 - a14 * a22;
  d6 = a13 * a24 - a14 * a23;

  u[0] = d6 - d5 + d4;
  u[1] = -d6 + d3 - d2;
  u[2] = d5 - d3 + d1;
  u[3] = -d4 + d2 - d1;
}

int Solver_plus::generate_direction_general(double *u, int n, bool new_working_set)
{
  int i, j, result, worknew;
  int n_minus_one = n - 1;
  int n_minus_two = n - 2;
  int i_minus_one;
  double *A0 = A[0];

  // compute A matrix
  if (new_working_set)
  {
    for (j = 0; j < n_minus_one; j++)
      A0[j] = 1;
  }

  double *Ai, *G, *G_old, *g, *g_old, *g_beta, *g_beta_old;
  int workj, start;

  if (new_working_set)
    start = 0;
  else
    start = n_minus_two;

  for (i = 1; i < n_minus_one; i++)
  {
    i_minus_one = i - 1;
    Ai = A[i];
    G = G_cg[i_minus_one];
    G_old = G_cg[i];
    g = g_cg[i_minus_one];
    g_old = g_cg[i];
    g_beta = g_beta_cg[i_minus_one];
    g_beta_old = g_beta_cg[i];

    for (j = start; j < n_minus_one; j++)
    {
      workj = work[j + 1];
      if (workj < l)
        Ai[j] = y[workj] * G[workj] + g[workj] / tau - y[workj] * G_old[workj] - g_old[workj] / tau;
      else
      {
        worknew = workj - l;
        Ai[j] = g_beta[worknew] / tau - g_beta_old[worknew] / tau;
      }
    }
  }

  // compute b vector
  if (new_working_set)
  {
    int work0 = work[0];
    b[0] = -1;
    if (work0 < l)
      for (i = 1; i <= n_minus_two; i++)
      {
        i_minus_one = i - 1;
        b[i] = -y[work0] * G_cg[i_minus_one][work0] - g_cg[i_minus_one][work0] / tau + y[work0] * G_cg[i][work0] + g_cg[i][work0] / tau;
      }
    else
    {
      worknew = work0 - l;
      for (i = 1; i <= n_minus_two; i++)
        b[i] = -g_beta_cg[i - 1][worknew] / tau + g_beta_cg[i][worknew] / tau;
    }
  }

  result = solve_linear_system(A, b, n_minus_one);
  if (result)
    return 1;
  u[0] = 1;
  memcpy(&u[1], b, sizeof_double * (n_minus_one));

  return 0;
}

void Solver_plus::generate_direction4y(double *u, bool new_working_set)
{
  int working3 = work[3];
  int y_i;
  int working_new;

  static int a11, a12, a13;
  int a14 = y[working3];

  double *G0 = G_cg[0];
  double *G1 = G_cg[1];
  double *g0 = g_cg[0];
  double *g1 = g_cg[1];
  double *g0beta = g_beta_cg[0];
  double *g1beta = g_beta_cg[1];

  static double a21, a22, a23, d1, d2, d4;
  double a24, d3, d5, d6;

  if (new_working_set)
  {
    int working0 = work[0];
    int working1 = work[1];
    int working2 = work[2];
    a11 = y[working0];
    a12 = y[working1];
    a13 = y[working2];

    if (working0 < l)
    {
      y_i = y[working0];
      a21 = y_i * G0[working0] + g0[working0] / tau - y_i * G1[working0] - g1[working0] / tau;
    }
    else
    {
      working_new = working0 - l;
      a21 = g0beta[working_new] / tau - g1beta[working_new] / tau;
    }

    if (working1 < l)
    {
      y_i = y[working1];
      a22 = y_i * G0[working1] + g0[working1] / tau - y_i * G1[working1] - g1[working1] / tau;
    }
    else
    {
      working_new = working1 - l;
      a22 = g0beta[working_new] / tau - g1beta[working_new] / tau;
    }

    if (working2 < l)
    {
      y_i = y[working2];
      a23 = y_i * G0[working2] + g0[working2] / tau - y_i * G1[working2] - g1[working2] / tau;
    }
    else
    {
      working_new = working2 - l;
      a23 = g0beta[working_new] / tau - g1beta[working_new] / tau;
    }

    d1 = a11 * a22 - a12 * a21;
    d2 = a11 * a23 - a13 * a21;
    d4 = a12 * a23 - a13 * a22;
  }

  if (working3 < l)
  {
    y_i = y[working3];
    a24 = y_i * G0[working3] + g0[working3] / tau - y_i * G1[working3] - g1[working3] / tau;
  }
  else
  {
    working_new = working3 - l;
    a24 = g0beta[working_new] / tau - g1beta[working_new] / tau;
  }

  d3 = a11 * a24 - a21 * a14;
  d5 = a12 * a24 - a14 * a22;
  d6 = a13 * a24 - a14 * a23;

  u[0] = d6 - d5 + d4;
  u[1] = -d6 + d3 - d2;
  u[2] = d5 - d3 + d1;
  u[3] = -d4 + d2 - d1;
}

void Solver_plus::generate_direction5y(double *u, bool new_working_set)
{
  int working4 = work[4];
  int working_new;
  int y_i;

  int y5 = y[working4];
  static int y1, y2, y3, y4;

  double *G0 = G_cg[0];
  double *G1 = G_cg[1];
  double *G2 = G_cg[2];

  double *g0 = g_cg[0];
  double *g1 = g_cg[1];
  double *g2 = g_cg[2];

  double *g0beta = g_beta_cg[0];
  double *g1beta = g_beta_cg[1];
  double *g2beta = g_beta_cg[2];

  double G0old, a15, a25;
  static double a11, a12, a13, a14, a21, a22, a23, a24;
  static double d12, d13, d14, d23, d24, d34, d123, d124, d134, d234;

  if (new_working_set)
  {
    int working0 = work[0];
    int working1 = work[1];
    int working2 = work[2];
    int working3 = work[3];
    y1 = y[working0];
    y2 = y[working1];
    y3 = y[working2];
    y4 = y[working3];

    if (working0 < l)
    {
      y_i = y[working0];
      G0old = y_i * G1[working0] + g1[working0] / tau;
      a11 = y_i * G0[working0] + g0[working0] / tau - G0old;
      a21 = G0old - y_i * G2[working0] - g2[working0] / tau;
    }
    else
    {
      working_new = working0 - l;
      G0old = g1beta[working_new] / tau;
      a11 = g0beta[working_new] / tau - G0old;
      a21 = G0old - g2beta[working_new] / tau;
    }

    if (working1 < l)
    {
      y_i = y[working1];
      G0old = y_i * G1[working1] + g1[working1] / tau;
      a12 = y_i * G0[working1] + g0[working1] / tau - G0old;
      a22 = G0old - y_i * G2[working1] - g2[working1] / tau;
    }
    else
    {
      working_new = working1 - l;
      G0old = g1beta[working_new] / tau;
      a12 = g0beta[working_new] / tau - G0old;
      a22 = G0old - g2beta[working_new] / tau;
    }

    if (working2 < l)
    {
      y_i = y[working2];
      G0old = y_i * G1[working2] + g1[working2] / tau;
      a13 = y_i * G0[working2] + g0[working2] / tau - G0old;
      a23 = G0old - y_i * G2[working2] - g2[working2] / tau;
    }
    else
    {
      working_new = working2 - l;
      G0old = g1beta[working_new] / tau;
      a13 = g0beta[working_new] / tau - G0old;
      a23 = G0old - g2beta[working_new] / tau;
    }

    if (working3 < l)
    {
      y_i = y[working3];
      G0old = y_i * G1[working3] + g1[working3] / tau;
      a14 = y_i * G0[working3] + g0[working3] / tau - G0old;
      a24 = G0old - y_i * G2[working3] - g2[working3] / tau;
    }
    else
    {
      working_new = working3 - l;
      G0old = g1beta[working_new] / tau;
      a14 = g0beta[working_new] / tau - G0old;
      a24 = G0old - g2beta[working_new] / tau;
    }

    d12 = a11 * a22 - a21 * a12;
    d13 = a11 * a23 - a13 * a21;
    d14 = a11 * a24 - a14 * a21;
    d23 = a12 * a23 - a13 * a22;
    d24 = a12 * a24 - a14 * a22;
    d34 = a13 * a24 - a14 * a23;
    d123 = y1 * d23 - y2 * d13 + y3 * d12;
    d124 = y1 * d24 - y2 * d14 + y4 * d12;
    d134 = y1 * d34 - y3 * d14 + y4 * d13;
    d234 = y2 * d34 - y3 * d24 + y4 * d23;
  }

  if (working4 < l)
  {
    y_i = y[working4];
    G0old = y_i * G1[working4] + g1[working4] / tau;
    a15 = y_i * G0[working4] + g0[working4] / tau - G0old;
    a25 = G0old - y_i * G2[working4] - g2[working4] / tau;
  }
  else
  {
    working_new = working4 - l;
    G0old = g1beta[working_new] / tau;
    a15 = g0beta[working_new] / tau - G0old;
    a25 = G0old - g2beta[working_new] / tau;
  }

  double d15 = a11 * a25 - a15 * a21;
  double d25 = a12 * a25 - a15 * a22;
  double d35 = a13 * a25 - a23 * a15;
  double d45 = a14 * a25 - a15 * a24;

  double d125 = y1 * d25 - y2 * d15 + y5 * d12;
  double d135 = y1 * d35 - y3 * d15 + y5 * d13;
  double d145 = y1 * d45 - y4 * d15 + y5 * d14;
  double d235 = y2 * d35 - y3 * d25 + y5 * d23;
  double d345 = y3 * d45 - y4 * d35 + y5 * d34;
  double d245 = y2 * d45 - y4 * d25 + y5 * d24;

  u[0] = d345 - d245 + d235 - d234;
  u[1] = -d345 + d145 - d135 + d134;
  u[2] = d245 - d145 + d125 - d124;
  u[3] = -d235 + d135 - d125 + d123;
  u[4] = d234 - d134 + d124 - d123;
}

int Solver_plus::generate_direction_y_general(double *u, int n, bool new_working_set)
{
  int i, j, result, worknew;
  double *A0 = A[0], *A1 = A[1];
  int n_minus_one = n - 1, n_minus_two = n - 2;
  int i_minus_one, i_minus_two, start;

  if (new_working_set)
    start = 0;
  else
    start = n_minus_two;

  // compute A matrix
  if (new_working_set)
    for (i = 0; i < n_minus_one; i++)
      A0[i] = 1;
  for (i = start; i < n_minus_one; i++)
    A1[i] = y[work[i + 1]];

  double *Ai, *G, *G_old, *g, *g_old, *g_beta, *g_beta_old;
  int workj, y_j, y_i;

  for (i = 2; i < n_minus_one; i++)
  {
    i_minus_one = i - 1;
    i_minus_two = i - 2;
    Ai = A[i];
    G = G_cg[i_minus_two];
    G_old = G_cg[i_minus_one];
    g = g_cg[i_minus_two];
    g_old = g_cg[i_minus_one];
    g_beta = g_beta_cg[i_minus_two];
    g_beta_old = g_beta_cg[i_minus_one];

    for (j = start; j < n_minus_one; j++)
    {
      workj = work[j + 1];
      if (workj < l)
      {
        y_j = y[workj];
        Ai[j] = y_j * G[workj] + g[workj] / tau - y_j * G_old[workj] - g_old[workj] / tau;
      }
      else
      {
        worknew = workj - l;
        Ai[j] = g_beta[worknew] / tau - g_beta_old[worknew] / tau;
      }
    }
  }

  // compute b vector
  if (new_working_set)
  {
    int work0 = work[0];
    b[0] = -1;
    b[1] = -y[work0];

    if (work0 < l)
      for (i = 2; i <= n_minus_two; i++)
      {
        i_minus_one = i - 1;
        i_minus_two = i - 2;
        y_i = y[work0];
        b[i] = -y_i * G_cg[i_minus_two][work0] - g_cg[i_minus_two][work0] / tau + y_i * G_cg[i_minus_one][work0] + g_cg[i_minus_one][work0] / tau;
      }
    else
    {
      worknew = work0 - l;
      for (i = 2; i <= n_minus_two; i++)
        b[i] = -g_beta_cg[i - 2][worknew] / tau + g_beta_cg[i - 1][worknew] / tau;
    }
  }

  result = solve_linear_system(A, b, n_minus_one);
  if (result)
    return 1;
  u[0] = 1;
  memcpy(&u[1], b, sizeof_double * (n_minus_one));

  return 0;
}

void Solver_plus::swap_index_alpha(int i, int j)
{
  Q->swap_index(i, j);
  Q_star->swap_index(i, j);
  swap(y[i], y[j]);
  swap(alpha_status[i], alpha_status[j]);
  swap(alpha[i], alpha[j]);
  swap(true_act_set[active_set[i]], true_act_set[active_set[j]]);
  swap(active_set[i], active_set[j]);
  swap(G[i], G[j]);
  swap(g[i], g[j]);
  swap(g_init[i], g_init[j]);
}

void Solver_plus::swap_index_beta(int i, int j)
{
  Q_star_beta->swap_index(i, j);
  swap(beta_status[i], beta_status[j]);
  swap(beta[i], beta[j]);
  swap(true_act_set_beta[active_set_beta[i]], true_act_set_beta[active_set_beta[j]]);
  swap(active_set_beta[i], active_set_beta[j]);
  swap(g_beta[i], g_beta[j]);
  swap(g_beta_init[i], g_beta_init[j]);
}

void Solver_plus::swap_index_alpha_cg(int i, int j)
{
  int k;
  Q->swap_index(i, j);
  Q_star->swap_index(i, j);
  swap(y[i], y[j]);
  swap(true_act_set[active_set[i]], true_act_set[active_set[j]]);
  swap(active_set[i], active_set[j]);
  swap(g_init[i], g_init[j]);

  for (k = 0; k < curr_depth + 1; k++)
  {
    swap(alpha_status_cg[k][i], alpha_status_cg[k][j]);
    swap(alpha_cg[k][i], alpha_cg[k][j]);
    swap(G_cg[k][i], G_cg[k][j]);
    swap(g_cg[k][i], g_cg[k][j]);
  }

  for (k = 0; k < working_set_size; k++)
  {
    if (i == work[k])
      work[k] = -1;
    else if (j == work[k])
      work[k] = i;
  }
  active[i] = active[j];
  active[j] = false;
}

void Solver_plus::swap_index_beta_cg(int i, int j)
{
  Q_star_beta->swap_index(i, j);
  swap(g_beta_init[i], g_beta_init[j]);
  swap(true_act_set_beta[active_set_beta[i]], true_act_set_beta[active_set_beta[j]]);
  swap(active_set_beta[i], active_set_beta[j]);

  int k;
  for (k = 0; k < curr_depth + 1; k++)
  {
    swap(beta_status_cg[k][i], beta_status_cg[k][j]);
    swap(beta_cg[k][i], beta_cg[k][j]);
    swap(g_beta_cg[k][i], g_beta_cg[k][j]);
  }

  int il = i + l;
  int jl = j + l;

  for (k = 0; k < working_set_size; k++)
  {
    if (il == work[k])
      work[k] = -1;
    else if (jl == work[k])
      work[k] = il;
  }
  active[il] = active[jl];
  active[jl] = false;
}

bool Solver_plus::do_shrinking_plus_cg()
{
  int i, j, y_i;
  double g_i, alpha_i, deriv_alpha_i;
  double max_B1 = -1e20, min_B1B2 = 1e20, max_A2 = -1e20, min_A2A4 = 1e20, max_A1 = -1e20, min_A1A3 = 1e20;
  bool done_shrinking;
  double *alpha, *G, *g, *g_beta, *beta;

  // compute all maxima and minima related to alphas
  alpha = alpha_cg[0];
  beta = beta_cg[0];
  g = g_cg[0];
  G = G_cg[0];
  g_beta = g_beta_cg[0];
  for (i = 0; i < active_size; i++)
  {
    alpha_i = alpha[i];
    g_i = g[i];
    y_i = y[i];
    deriv_alpha_i = y_i * G[i] + g_i / tau;

    // max A2
    if (alpha_i > 1e-8 && y_i == -1 && deriv_alpha_i > max_A2)
      max_A2 = deriv_alpha_i;

    // min A2A4
    if (y_i == -1 && deriv_alpha_i < min_A2A4)
      min_A2A4 = deriv_alpha_i;

    // max A1
    if (alpha_i > 1e-8 && y_i == 1 && deriv_alpha_i > max_A1)
      max_A1 = deriv_alpha_i;

    // min A1A3max_A2, min_A2A4, max_A1, min_A1A3
    if (y_i == 1 && deriv_alpha_i < min_A1A3)
      min_A1A3 = deriv_alpha_i;
  }

  // compute all maxima and minima related to betas
  for (i = 0; i < active_size_beta; i++)
  {
    g_i = g_beta[i];

    // max B1
    if (beta[i] > 1e-8 && g_i > max_B1)
      max_B1 = g_i;

    // min B1B2
    if (g_i < min_B1B2)
      min_B1B2 = g_i;
  }

  max_B1 /= tau;
  min_B1B2 /= tau;

  if (unshrink == false && max_B1 - min_B1B2 < eps * 10 &&
      max_A2 - min_A2A4 < eps * 10 && max_A1 - min_A1A3 < eps * 10 &&
      2 * max_B1 + 2 - min_A1A3 - min_A2A4 < eps * 10 && max_A1 + max_A2 - 2 * min_B1B2 - 2 < eps * 10)
  {
    unshrink = true;
    reconstruct_gradient_plus_cg();
    active_size = l;
    active_size_beta = l;
  }

  if (active_size_beta > 2)
  {
    for (i = 0; i < active_size_beta; i++)
    {
      if (active_size_beta <= 2)
        break;
      if (be_shrunk_beta_cg(i, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
      {
        active_size_beta--;
        //        fprintf(stdout,"shrinking beta %d\n",i);
        //fflush(stdout);
        if (active_size_beta == i)
        {
          if (active[i + l])
          {
            active[i + l] = false;
            for (j = 0; j < working_set_size; j++)
              if (work[j] == i + l)
              {
                work[j] = -1;
                break;
              }
          }
        }
        else
        {
          while (active_size_beta > i)
          {
            if (!be_shrunk_beta_cg(active_size_beta, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
            {
              swap_index_beta_cg(i, active_size_beta);
              done_shrinking = true;
              break;
            }
            else
            {
              if (active[active_size_beta + l])
              {
                active[active_size_beta + l] = false;
                for (j = 0; j < working_set_size; j++)
                  if (work[j] == active_size_beta + l)
                  {
                    work[j] = -1;
                    break;
                  }
              }
              active_size_beta--;
              if (active_size_beta <= 2)
                break;
            }
          }
        }
      }
    }
  }

  if (active_size > 2)
  {
    for (i = 0; i < active_size; i++)
    {
      if (active_size <= 2)
        break;
      if (be_shrunk_alpha_cg(i, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
      {
        active_size--;
        if (active_size == i)
        {
          if (active[i])
          {
            active[i] = false;
            for (j = 0; j < working_set_size; j++)
              if (work[j] == i)
              {
                work[j] = -1;
                break;
              }
          }
        }
        else
        {
          while (active_size > i)
          {
            if (!be_shrunk_alpha_cg(active_size, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
            {
              swap_index_alpha_cg(i, active_size);
              done_shrinking = true;
              break;
            }
            else
            {
              if (active[active_size])
              {
                active[active_size] = false;
                for (j = 0; j < working_set_size; j++)
                  if (work[j] == active_size)
                  {
                    work[j] = -1;
                    break;
                  }
              }
              active_size--;
            }
          }
        }
      }
    }
  }

  for (i = 0; i < working_set_size; i++)
    if (work[i] == -1)
    {
      working_set_size--;
      while (working_set_size > i)
      {
        if (work[working_set_size] != -1)
        {
          work[i] = work[working_set_size];
          work[working_set_size] = -1;
          break;
        }
        working_set_size--;
      }
    }

  switch (working_set_type)
  {
  case ALPHAS:
  case BETAS:
    if (working_set_size <= 1)
    {
      working_set_size = 0;
      curr_depth = 0;
    }
    break;
  case ALPHAS_BETAS:
    if (working_set_size <= 2)
    {
      working_set_size = 0;
      curr_depth = 0;
    }
    break;
  case ALPHAS_DIFF_SIGN:
    if (working_set_size <= 3)
    {
      working_set_size = 0;
      curr_depth = 0;
    }
  }

  return done_shrinking;
}

void Solver_plus::reconstruct_gradient_plus_cg()
{
  int i, j, k, true_i, act_set_i;
  double *alpha_k, *beta_k, *G_k, *g_k, *g_beta_k;

  //  fprintf(stdout,"reconstructing gradient\n");
  //fflush(stdout);

  if (active_size < l)
  {
    for (k = 0; k < curr_depth + 1; k++)
    {
      alpha_k = alpha_cg[k];
      beta_k = beta_cg[k];
      G_k = G_cg[k];
      g_k = g_cg[k];
      for (i = active_size; i < l; i++)
      {
        const Qfloat *Q_i = Q->get_Q(i, l);
        const Qfloat *Q_i_star = Q_star->get_Q(i, l);

        true_i = active_set[i];
        act_set_i = true_act_set_beta[true_i];

        const Qfloat *Q_i_star_beta = Q_star_beta->get_Q(act_set_i, l);
        G_k[i] = 0;
        g_k[i] = g_init[i];
        for (j = 0; j < l; j++)
          if (alpha_k[j] > 1e-8)
          {
            G_k[i] += alpha_k[j] * y[j] * Q_i[j];
            g_k[i] += alpha_k[j] * Q_i_star[j];
          }
        for (j = 0; j < l; j++)
          if (beta_k[j] > 1e-8)
            g_k[i] += beta_k[j] * Q_i_star_beta[j];
      }
    }
  }

  if (active_size_beta < l)
  {
    for (k = 0; k < curr_depth + 1; k++)
    {
      alpha_k = alpha_cg[k];
      beta_k = beta_cg[k];
      g_beta_k = g_beta_cg[k];
      for (i = active_size_beta; i < l; i++)
      {
        const Qfloat *Q_i_star_beta = Q_star_beta->get_Q(i, l);

        true_i = active_set_beta[i];
        act_set_i = true_act_set[true_i];
        const Qfloat *Q_i_star = Q_star->get_Q(act_set_i, l);

        g_beta_k[i] = g_beta_init[i];

        for (j = 0; j < l; j++)
          if (beta_k[j] > 1e-8)
            g_beta_k[i] += beta_k[j] * Q_i_star_beta[j];

        for (j = 0; j < l; j++)
          if (alpha_k[j] > 1e-8)
            g_beta_k[i] += alpha_k[j] * Q_i_star[j];
      }
    }
  }
}

int Solver_plus::wss_first_order(double *best_u, double &lambda_star)
{
  int i, y_i;
  double g_i, alpha_i, deriv_alpha_i;
  double max_B1 = -1e20, min_B1B2 = 1e20, max_A2 = -1e20, min_A2A4 = 1e20, max_A1 = -1e20, min_A1A3 = 1e20;
  double *alpha, *G, *g, *g_beta, *beta;
  int best_B1 = -1, best_B1B2 = -1, best_A2 = -1, best_A2A4 = -1, best_A1 = -1, best_A1A3 = -1;
  int type_selected[3], selected_indices[3][3], alpha_beta_type = -1;

  for (i = 0; i < working_set_size; i++)
  {
    active[work[i]] = false;
    work[i] = -1;
  }
  working_set_size = 0;
  curr_depth = 0;

  // compute all maxima and minima related to alphas
  alpha = alpha_cg[0];
  beta = beta_cg[0];
  g = g_cg[0];
  G = G_cg[0];
  g_beta = g_beta_cg[0];
  for (i = 0; i < active_size; i++)
  {
    alpha_i = alpha[i];
    g_i = g[i];
    y_i = y[i];
    deriv_alpha_i = y_i * G[i] + g_i / tau;

    // max A2
    if (alpha_i > 1e-8 && y_i == -1 && deriv_alpha_i > max_A2)
    {
      max_A2 = deriv_alpha_i;
      best_A2 = i;
    }

    // min A2A4
    if (y_i == -1 && deriv_alpha_i < min_A2A4)
    {
      min_A2A4 = deriv_alpha_i;
      best_A2A4 = i;
    }

    // max A1
    if (alpha_i > 1e-8 && y_i == 1 && deriv_alpha_i > max_A1)
    {
      max_A1 = deriv_alpha_i;
      best_A1 = i;
    }

    // min A1A3max_A2, min_A2A4, max_A1, min_A1A3
    if (y_i == 1 && deriv_alpha_i < min_A1A3)
    {
      min_A1A3 = deriv_alpha_i;
      best_A1A3 = i;
    }
  }

  // compute all maxima and minima related to betas
  for (i = 0; i < active_size_beta; i++)
  {
    g_i = g_beta[i];

    // max B1
    if (beta[i] > 1e-8 && g_i > max_B1)
    {
      max_B1 = g_i;
      best_B1 = i;
    }

    // min B1B2
    if (g_i < min_B1B2)
    {
      min_B1B2 = g_i;
      best_B1B2 = i;
    }
  }

  double gap[3];

  for (i = 0; i < 3; i++)
    gap[i] = -1;

  max_B1 /= tau;
  min_B1B2 /= tau;

  // select maximal violating pairs
  if (max_B1 - min_B1B2 < eps)
    type_selected[0] = 0;
  else
  {
    type_selected[0] = 1;
    selected_indices[0][0] = best_B1;
    selected_indices[0][1] = best_B1B2;
    gap[0] = max_B1 - min_B1B2;
  }

  if (max_A2 - min_A2A4 < eps && max_A1 - min_A1A3 < eps)
    type_selected[1] = 0;
  else
  {
    type_selected[1] = 1;
    if (max_A2 - min_A2A4 > max_A1 - min_A1A3)
    {
      selected_indices[1][0] = best_A2;
      selected_indices[1][1] = best_A2A4;
      gap[1] = max_A2 - min_A2A4;
    }
    else
    {
      selected_indices[1][0] = best_A1;
      selected_indices[1][1] = best_A1A3;
      gap[1] = max_A1 - min_A1A3;
    }
  }

  if (2 * max_B1 + 2 - min_A1A3 - min_A2A4 < eps && max_A1 + max_A2 - 2 * min_B1B2 - 2 < eps)
    type_selected[2] = 0;
  else
  {
    type_selected[2] = 1;
    if (2 * max_B1 + 2 - min_A1A3 - min_A2A4 > max_A1 + max_A2 - 2 * min_B1B2 - 2)
    {
      selected_indices[2][0] = best_A1A3;
      selected_indices[2][1] = best_A2A4;
      selected_indices[2][2] = best_B1;
      gap[2] = 2 * max_B1 + 2 - min_A1A3 - min_A2A4;
      alpha_beta_type = 0;
    }
    else
    {
      selected_indices[2][0] = best_A1;
      selected_indices[2][1] = best_A2;
      selected_indices[2][2] = best_B1B2;
      gap[2] = max_A1 + max_A2 - 2 * min_B1B2 - 2;
      alpha_beta_type = 1;
    }
  }

  if (type_selected[0] + type_selected[1] + type_selected[2] == 0)
    return 1;

  if (gap[2] >= gap[1] && gap[2] >= gap[0])
  {
    working_set_type = ALPHAS_BETAS;
    working_set_size = 3;
    work[0] = selected_indices[2][0];
    work[1] = selected_indices[2][1];
    work[2] = selected_indices[2][2] + l;
    active[work[0]] = true;
    active[work[1]] = true;
    active[work[2]] = true;
  }
  else if (gap[1] >= gap[0])
  {
    working_set_type = ALPHAS;
    working_set_size = 2;
    work[0] = selected_indices[1][0];
    work[1] = selected_indices[1][1];
    active[work[0]] = true;
    active[work[1]] = true;
  }
  else
  {
    working_set_type = BETAS;
    working_set_size = 2;
    work[0] = selected_indices[0][0] + l;
    work[1] = selected_indices[0][1] + l;
    active[work[0]] = true;
    active[work[1]] = true;
  }

  Qfloat *Q_i, *Q_i_star, *Q_i_star_beta, *Q_j_star_beta, *Q_j, *Q_j_star, *Q_k_star, *Q_k_star_beta;
  int work0 = work[0], work1 = work[1], work2, true_k, act_set_k;

  switch (working_set_type)
  {
  case BETAS:
    best_u[0] = -1;
    best_u[1] = 1;
    work0 -= l;
    work1 -= l;
    Q_i_star_beta = Q_star_beta->get_Q(work0, active_size_beta);
    Q_j_star_beta = Q_star_beta->get_Q(work1, active_size_beta);
    lambda_star = (g_beta[work0] - g_beta[work1]) / (Q_i_star_beta[work0] + Q_j_star_beta[work1] - 2 * Q_i_star_beta[work1]);
    lambda_star = min(lambda_star, beta[work0]);
    break;

  case ALPHAS:
    best_u[0] = -1;
    best_u[1] = 1;
    Q_i = Q->get_Q(work0, active_size);
    Q_j = Q->get_Q(work1, active_size);
    Q_i_star = Q_star->get_Q(work0, active_size);
    Q_j_star = Q_star->get_Q(work1, active_size);
    lambda_star = y[work0] * G[work0] - y[work1] * G[work1] + (g[work0] - g[work1]) / tau;
    lambda_star /= Q_i[work0] + Q_j[work1] - 2 * Q_i[work1] + (Q_i_star[work0] + Q_j_star[work1] - 2 * Q_i_star[work1]) / tau;
    lambda_star = min(lambda_star, alpha[work[0]]);
    break;

  case ALPHAS_BETAS:
    work2 = work[2] - l;
    Q_i = Q->get_Q(work0, active_size);
    Q_j = Q->get_Q(work1, active_size);
    Q_i_star = Q_star->get_Q(work0, active_size);
    Q_j_star = Q_star->get_Q(work1, active_size);
    Q_k_star_beta = Q_star_beta->get_Q(work2, active_size_beta);
    true_k = active_set_beta[work2];
    act_set_k = true_act_set[true_k];
    Q_k_star = Q_star->get_Q(act_set_k, active_size);
    lambda_star = y[work0] * G[work0] + y[work1] * G[work1] - 2 + (g[work0] + g[work1] - 2 * g_beta[work2]) / tau;
    lambda_star /= Q_i[work0] + Q_j[work1] - 2 * Q_i[work1] + (Q_i_star[work0] + Q_j_star[work1] + 2 * Q_i_star[work1] - 4 * Q_k_star[work0] - 4 * Q_k_star[work1] + 4 * Q_k_star_beta[work2]) / tau;

    if (alpha_beta_type == 0)
    {
      best_u[0] = 1;
      best_u[1] = 1;
      best_u[2] = -2;
      lambda_star = min(-lambda_star, beta[work2] / 2);
    }
    else
    {
      best_u[0] = -1;
      best_u[1] = -1;
      best_u[2] = 2;
      lambda_star = min(lambda_star, min(alpha[work0], alpha[work1]));
    }
  }

  return 0;
}

bool Solver_plus::feasible_direction(double *u)
{
  int i, work_i;
  int new_working_set_size = working_set_size + 1;
  double *alpha = alpha_cg[0];
  double *beta = beta_cg[0];

  for (i = 0; i < new_working_set_size; i++)
  {
    work_i = work[i];
    if (work_i < l)
    {
      if (alpha[work_i] <= 1e-8 && u[i] < 0)
        return false;
    }
    else
    {
      if (beta[work_i - l] <= 1e-8 && u[i] < 0)
        return false;
    }
  }

  return true;
}

bool Solver_plus::compute_gain(int new_working_set_size, double &gain, double *best_u, double &lambda, bool new_working_set)
{

  //info();
  //printf("Solver_Plus::compute_gain\n");
  bool result, negligible, feasible_u, feasible_minus_u;
  double nominator, nominator1, nominator2, denominator, ui;
  int i, j, k, worki, workj, true_j, y_i;
  Qfloat *Q_i, *Q_i_star;
  //double u[new_working_set_size], minus_u[new_working_set_size];
  // a hack to bypass MSVC compiler restriction <Okba BEKHELIFI>
  // double u[20], minus_u[20];
  double u[5], minus_u[5];
  double *G, *g, *g_beta, *alpha, *beta;
  double tmp1, tmp2;
  double best_u_i, best_u_j;
  double u_i;

  G = G_cg[0];
  g = g_cg[0];
  g_beta = g_beta_cg[0];
  alpha = alpha_cg[0];
  beta = beta_cg[0];

  if (working_set_type == ALPHAS_BETAS || working_set_type == ALPHAS_DIFF_SIGN)
    result = generate_direction_y(u, new_working_set_size, new_working_set);
  else
    result = generate_direction(u, new_working_set_size, new_working_set);
  if (result)
    return false;

  for (i = 0; i < new_working_set_size; i++)
  {
    u_i = u[i];
    if (u_i == 0)
      return false;
    minus_u[i] = -u_i;
  }

  feasible_u = feasible_direction(u);
  feasible_minus_u = feasible_direction(minus_u);

  if (!feasible_u && !feasible_minus_u)
    return false;

  // choose which of {u,minus_u} is a descent direction
  nominator1 = 0;
  nominator2 = 0;
  nominator = 0;
  for (i = 0; i < new_working_set_size; i++)
  {
    worki = work[i];
    if (worki < l)
    {
      u_i = u[i];
      tmp1 = (y[worki] * G[worki] - 1) * u_i;
      tmp2 = g[worki] * u_i;
      nominator += tmp1 + tmp2 / tau;
      nominator1 += tmp1;
      nominator2 += tmp2;
    }
    else
    {
      tmp1 = g_beta[worki - l] * u[i];
      nominator += tmp1 / tau;
      nominator2 += tmp1;
    }
  }

  if (fabs(nominator1) < eps && fabs(nominator2) < eps)
    return false;

  if (fabs(nominator) < eps)
    return false;

  if (nominator < 0)
    if (feasible_u)
      memcpy(best_u, u, sizeof_double * new_working_set_size);
    else
      return false;
  else if (feasible_minus_u)
  {
    nominator *= -1;
    memcpy(best_u, minus_u, sizeof_double * new_working_set_size);
  }
  else
    return false;

  // compute denominator
  denominator = 0;
  for (i = 0; i < new_working_set_size; i++)
  {
    worki = work[i];
    best_u_i = best_u[i];
    y_i = y[worki];
    if (worki < l)
    {
      Q_i = Q->get_Q(worki, active_size);
      Q_i_star = Q_star->get_Q(worki, l); //active_size);
      denominator += best_u_i * best_u_i * (QD[worki] + QD_star[worki] / tau);
      for (j = i + 1; j < new_working_set_size; j++)
      {
        workj = work[j];
        best_u_j = best_u[j];
        if (workj < l)
          denominator += 2 * best_u_i * best_u_j * (y_i * y[workj] * Q_i[workj] + Q_i_star[workj] / tau);
        else
        {
          k = active_set_beta[workj - l];
          true_j = true_act_set[k];
          denominator += 2 * best_u_i * best_u_j * Q_i_star[true_j] / tau;
        }
      }
    }
    else
    {
      worki -= l;
      Q_i_star = Q_star_beta->get_Q(worki, l); //active_size_beta);
      denominator += best_u_i * best_u_i * QD_star_beta[worki] / tau;
      for (j = i + 1; j < new_working_set_size; j++)
      {
        workj = work[j];
        best_u_j = best_u[j];
        if (workj < l)
        {
          k = active_set[workj];
          true_j = true_act_set_beta[k];
          denominator += 2 * best_u_i * best_u_j * Q_i_star[true_j] / tau;
        }
        else
          denominator += 2 * best_u_i * best_u_j * Q_i_star[workj - l] / tau;
      }
    }
  }

  if (denominator <= 0)
    denominator = TAU;

  lambda = -nominator / denominator;
  for (i = 0; i < new_working_set_size; i++)
  {
    ui = best_u[i];
    worki = work[i];
    if (ui > 0)
      if (worki < l)
        lambda = max(lambda, -alpha[worki] / ui);
      else
        lambda = max(lambda, -beta[worki - l] / ui);
    else if (worki < l)
      lambda = min(lambda, -alpha[worki] / ui);
    else
      lambda = min(lambda, -beta[worki - l] / ui);
  }

  if (fabs(lambda) < 1e-3)
    return false;
  negligible = false;
  for (i = 0; i < new_working_set_size; i++)
    if (fabs(lambda * best_u[i]) < 1e-3)
    {
      negligible = true;
      break;
    }
  if (negligible)
    return false;

  gain = nominator * nominator / (2 * denominator);
  return true;
}

int Solver_plus::select_working_set_plus_incrementally(double *best_u, double &lambda_star, double &best_gain)
{
  int k, best_k, new_working_set_size = working_set_size + 1, n_betas = 0, n_pos = 0, n_neg = 0;
  double gain, lambda;
  double *curr_u;
  int work_k, work0, y_k, kl;
  int new_working_set_size_bytes = sizeof_double * new_working_set_size;
  bool chose_alpha = false, chose_beta = false;

  curr_u = new double[new_working_set_size];
  best_gain = 0;

  if (working_set_type == ALPHAS_BETAS)
  {
    for (k = 0; k < working_set_size; k++)
    {
      work_k = work[k];
      if (work_k < l)
      {
        if (y[work_k] == -1)
          n_neg++;
        else
          n_pos++;
      }
      else
        n_betas++;
    }
  }

  if (working_set_type == ALPHAS || (working_set_type == ALPHAS_BETAS && (n_betas > 0 || n_pos + n_neg >= 3)) || working_set_type == ALPHAS_DIFF_SIGN)
  {

    work0 = work[0];
    for (k = 0; k < active_size; k++)
    {

      y_k = y[k];
      if (working_set_type == ALPHAS && y_k != y[work0])
        continue;

      if (working_set_type == ALPHAS_BETAS)
      {
        if (n_pos == 0 && y_k == -1)
          continue;

        if (n_neg == 0 && y_k == 1)
          continue;

        if (n_betas == 0)
        {
          if (n_pos == 1 && y_k == -1)
            continue;
          if (n_neg == 1 && y_k == 1)
            continue;
        }
      }

      if (working_set_type == ALPHAS_DIFF_SIGN)
      {
        if (n_pos == 1 && y_k == -1)
          continue;
        if (n_pos == -1 && y_k == 1)
          continue;
      }

      // check that k is not in the working set
      if (active[k])
        continue;

      // generate direction
      work[working_set_size] = k;

      if (!compute_gain(new_working_set_size, gain, curr_u, lambda, k == 0))
        continue;

      if (gain > best_gain)
      {
        chose_alpha = true;
        best_gain = gain;
        best_k = k;
        memcpy(best_u, curr_u, new_working_set_size_bytes);
        lambda_star = lambda;
      }
    }
  }

  if (working_set_type == BETAS || (working_set_type == ALPHAS_BETAS && n_pos > 0 && n_neg > 0) || working_set_type == ALPHAS_DIFF_SIGN)
    for (k = 0; k < active_size_beta; k++)
    {

      kl = k + l;
      // check that k is not in the working set
      if (active[kl])
        continue;

      // generate direction
      work[working_set_size] = kl;

      if (!compute_gain(new_working_set_size, gain, curr_u, lambda, best_gain == 0 && k == 0))
        continue;

      if (gain > best_gain)
      {
        chose_beta = true;
        chose_alpha = false;
        best_gain = gain;
        best_k = kl;
        memcpy(best_u, curr_u, new_working_set_size_bytes);
        lambda_star = lambda;
      }
    }

  if (best_gain == 0)
  {
    delete[] curr_u;
    return 1;
  }

  work[working_set_size] = best_k;
  active[best_k] = true;
  working_set_size++;

  if (working_set_type == ALPHAS_DIFF_SIGN && chose_beta)
    working_set_type = ALPHAS_BETAS;
  else if (working_set_type == ALPHAS_BETAS && chose_alpha && (n_betas == 0))
    working_set_type = ALPHAS_DIFF_SIGN;

  delete[] curr_u;
  return 0;
}

int Solver_plus::select_working_set_plus_hmg(double *best_u, double &lambda_star, double &incr_gain)
{
  int i;

  if (prev_depth < max_depth)
    return select_working_set_plus_incrementally(best_u, lambda_star, incr_gain);

  // throw away the oldest active example and select the working set incrementally
  int working_set_size_minus_one = working_set_size - 1;
  active[work[0]] = false;
  for (i = 0; i < working_set_size_minus_one; i++)
    work[i] = work[i + 1];
  work[working_set_size - 1] = -1;
  working_set_size--;

  return select_working_set_plus_incrementally(best_u, lambda_star, incr_gain);
}

// return 1 if already optimal, return 0 otherwise
int Solver_plus::select_working_set_plus_hmg2(double *best_u, double &lambda_star, double &absolute_best_z)
{
  int i, j, best_B1 = -1, best_B1B2 = -1, best_A2 = -1, best_A2A4 = -1, best_A1 = -1, best_A1A3 = -1, i_ind, j_ind, k_ind;
  int type_selected[3], selected_indices[3][3];
  double max_B1 = -1e20, min_B1B2 = 1e20, max_A2 = -1e20, min_A2A4 = 1e20, max_A1 = -1e20, min_A1A3 = 1e20;
  double alpha_i, G_i, g_i, g_j, y_i, y_j, deriv_alpha_i, first_order_criterion;
  double max_z[3], z, nominator, nominator_base, denominator, denominator_base, j_deriv, Q_star_ii;
  int best_z_index[3], true_k, act_set_k;
  Qfloat *Q_i, *Q_i_star, *Q_k_star, *Q_k_star_beta;
  double *alpha, *G, *g, *g_beta, *beta, lambda[3];
  double nominator_base1, nominator_base2, j_deriv1, j_deriv2, nominator1, nominator2;
  double max_A2_1, max_A2_2, min_A2A4_1, min_A2A4_2, max_A1_1, max_A1_2, min_A1A3_1, min_A1A3_2;
  double deriv_alpha_i1, deriv_alpha_i2;

  absolute_best_z = 0;
  // first-order working set selection

  for (i = 0; i < working_set_size; i++)
  {
    active[work[i]] = false;
    work[i] = -1;
  }
  working_set_size = 0;
  curr_depth = 0;

  // compute all maxima and minima related to alphas
  alpha = alpha_cg[0];
  beta = beta_cg[0];
  g = g_cg[0];
  G = G_cg[0];
  g_beta = g_beta_cg[0];
  for (i = 0; i < active_size; i++)
  {
    alpha_i = alpha[i];
    G_i = G[i];
    g_i = g[i];
    y_i = y[i];
    deriv_alpha_i1 = y_i * G[i];
    deriv_alpha_i2 = g_i;
    deriv_alpha_i = deriv_alpha_i1 + deriv_alpha_i2 / tau;

    // max A2
    if (alpha_i > 1e-8 && y_i == -1 && deriv_alpha_i > max_A2)
    {
      max_A2 = deriv_alpha_i;
      best_A2 = i;
      max_A2_1 = deriv_alpha_i1;
      max_A2_2 = deriv_alpha_i2;
    }

    // min A2A4
    if (y_i == -1 && deriv_alpha_i < min_A2A4)
    {
      min_A2A4 = deriv_alpha_i;
      best_A2A4 = i;
      min_A2A4_1 = deriv_alpha_i1;
      min_A2A4_2 = deriv_alpha_i2;
    }

    // max A1
    if (alpha_i > 1e-8 && y_i == 1 && deriv_alpha_i > max_A1)
    {
      max_A1 = deriv_alpha_i;
      best_A1 = i;
      max_A1_1 = deriv_alpha_i1;
      max_A1_2 = deriv_alpha_i2;
    }

    // min A1A3max_A2, min_A2A4, max_A1, min_A1A3
    if (y_i == 1 && deriv_alpha_i < min_A1A3)
    {
      min_A1A3 = deriv_alpha_i;
      min_A1A3_1 = deriv_alpha_i1;
      min_A1A3_2 = deriv_alpha_i2;
      best_A1A3 = i;
    }
  }

  // compute all maxima and minima related to betas
  for (i = 0; i < active_size_beta; i++)
  {
    g_i = g_beta[i];

    // max B1
    if (beta[i] > 1e-8 && g_i > max_B1)
    {
      max_B1 = g_i;
      best_B1 = i;
    }

    // min B1B2
    if (g_i < min_B1B2)
    {
      min_B1B2 = g_i;
      best_B1B2 = i;
    }
  }

  max_B1 /= tau;
  min_B1B2 /= tau;

  // select maximal violating pairs
  if (max_B1 - min_B1B2 < eps)
    type_selected[0] = 0;
  else
  {
    type_selected[0] = 1;
    selected_indices[0][0] = best_B1;
    selected_indices[0][1] = best_B1B2;
  }

  if (((max_A2 - min_A2A4 < eps) || ((max_A2_1 - min_A2A4_1 < eps) && (max_A2_2 - min_A2A4_2 < eps))) &&
      ((max_A1 - min_A1A3 < eps) || ((max_A1_1 - min_A1A3_1 < eps) && (max_A1_2 - min_A1A3_2 < eps))))
    type_selected[1] = 0;
  else
  {
    if ((max_A2 - min_A2A4 > max_A1 - min_A1A3) && ((max_A2_1 - min_A2A4_1 >= eps) || (max_A2_2 - min_A2A4_2 >= eps)))
    {
      type_selected[1] = 1;
      selected_indices[1][0] = best_A2;
      selected_indices[1][1] = best_A2A4;
    }
    else
    {
      if ((max_A2 - min_A2A4 <= max_A1 - min_A1A3) && ((max_A1_1 - min_A1A3_1 >= eps) || (max_A1_2 - min_A1A3_2 >= eps)))
      {
        type_selected[1] = 1;
        selected_indices[1][0] = best_A1;
        selected_indices[1][1] = best_A1A3;
      }
      else
        type_selected[1] = 0;
    }
  }

  if (((2 * max_B1 + 2 - min_A1A3 - min_A2A4 < eps) || ((2 - min_A1A3_1 - min_A2A4_1 < eps) && (2 * max_B1 * tau - min_A1A3_2 - min_A2A4_2 < eps))) &&
      ((max_A1 + max_A2 - 2 * min_B1B2 - 2 < eps) || ((max_A1_1 + max_A2_1 - 2 < eps) && (max_A1_2 + max_A2_2 - 2 * min_B1B2 * tau < eps))))
    type_selected[2] = 0;
  else
  {
    if ((2 * max_B1 + 2 - min_A1A3 - min_A2A4 > max_A1 + max_A2 - 2 * min_B1B2 - 2) && ((2 - min_A1A3_1 - min_A2A4_1 >= eps) || (2 * max_B1 * tau - min_A1A3_2 - min_A2A4_2 >= eps)))
    {
      type_selected[2] = 1;
      selected_indices[2][0] = best_A1A3;
      selected_indices[2][1] = best_A2A4;
      selected_indices[2][2] = best_B1;
    }
    else
    {
      if ((2 * max_B1 + 2 - min_A1A3 - min_A2A4 <= max_A1 + max_A2 - 2 * min_B1B2 - 2) && ((max_A1_1 + max_A2_1 - 2 >= eps) || (max_A1_2 + max_A2_2 - 2 * min_B1B2 * tau >= eps)))
      {
        type_selected[2] = 1;
        selected_indices[2][0] = best_A1;
        selected_indices[2][1] = best_A2;
        selected_indices[2][2] = best_B1B2;
      }
      else
        type_selected[2] = 0;
    }
  }

  if (type_selected[0] + type_selected[1] + type_selected[2] == 0)
    return 1;

  for (i = 0; i < 3; i++)
    max_z[i] = -1e20;

  // second-order working set selection
  if (type_selected[0] == 1)
  {
    i_ind = selected_indices[0][0];
    g_i = g_beta[i_ind];
    Q_i_star = Q_star_beta->get_Q(i_ind, active_size_beta);
    Q_star_ii = Q_i_star[i_ind];
    for (j = 0; j < active_size_beta; j++)
    {
      g_j = g_beta[j];
      if (eps + g_j / tau < g_i / tau)
      {
        nominator = g_i - g_j;
        denominator = Q_star_ii + QD_star_beta[j] - 2 * Q_i_star[j];
        z = nominator * nominator / (2 * tau * denominator);
        if (z > max_z[0])
        {
          max_z[0] = z;
          best_z_index[0] = j;
          lambda[0] = nominator / denominator;
        }
      }
    }
  }

  if (type_selected[1] == 1)
  {
    i_ind = selected_indices[1][0];
    y_i = y[i_ind];
    Q_i = Q->get_Q(i_ind, active_size);
    Q_i_star = Q_star->get_Q(i_ind, active_size);
    nominator_base = y_i * G[i_ind] + g[i_ind] / tau;
    nominator_base1 = y_i * G[i_ind];
    nominator_base2 = g[i_ind];
    denominator_base = 2 * (Q_i[i_ind] + Q_i_star[i_ind] / tau);

    for (j = 0; j < active_size; j++)
    {
      y_j = y[j];
      j_deriv = y_j * G[j] + g[j] / tau;
      j_deriv1 = y_j * G[j];
      j_deriv2 = g[j];

      if (y_j == y_i && j_deriv + eps < nominator_base && ((j_deriv1 + eps < nominator_base1) || (j_deriv2 + eps < nominator_base2)))
      {
        j_deriv = j_deriv1 + j_deriv2 / tau;
        nominator = nominator_base - j_deriv;
        denominator = denominator_base + 2 * (QD[j] - 2 * Q_i[j] + (QD_star[j] - 2 * Q_i_star[j]) / tau);
        z = nominator * nominator / denominator;
        if (z > max_z[1])
        {
          max_z[1] = z;
          best_z_index[1] = j;
          lambda[1] = nominator / (denominator / 2);
        }
      }
    }
  }

  if (type_selected[2] == 1)
  {
    i_ind = selected_indices[2][0];
    j_ind = selected_indices[2][1];
    k_ind = selected_indices[2][2];
    Q_i = Q->get_Q(i_ind, active_size);
    Q_i_star = Q_star->get_Q(i_ind, active_size);
    Q_k_star_beta = Q_star_beta->get_Q(k_ind, active_size_beta);

    true_k = active_set_beta[k_ind];
    act_set_k = true_act_set[true_k];
    Q_k_star = Q_star->get_Q(act_set_k, active_size);

    nominator_base = y[i_ind] * G[i_ind] - 2 + (g[i_ind] - 2 * g_beta[k_ind]) / tau;
    nominator_base1 = y[i_ind] * G[i_ind] - 2;
    nominator_base2 = g[i_ind] - 2 * g_beta[k_ind];
    denominator_base = 2 * (Q_i[i_ind] + (Q_i_star[i_ind] - 4 * Q_k_star[i_ind] + 4 * Q_k_star_beta[k_ind]) / tau);
    first_order_criterion = nominator_base + y[j_ind] * G[j_ind] + g[j_ind] / tau;
    for (j = 0; j < active_size; j++)
    {
      if (y[j] == -1)
      {
        nominator1 = nominator_base1 + y[j] * G[j];
        nominator2 = nominator_base2 + g[j];
        nominator = nominator_base + y[j] * G[j] + g[j] / tau;
        if ((first_order_criterion < 0 && nominator < -eps && ((nominator1 < -eps) || (nominator2 < -eps))) ||
            (first_order_criterion > 0 && alpha[j] > 1e-8 && nominator > eps && ((nominator1 > eps) || (nominator2 > eps))))
        {
          denominator = denominator_base + 2 * (QD[j] - 2 * Q_i[j] + (QD_star[j] + 2 * Q_i_star[j] - 4 * Q_k_star[j]) / tau);
          z = nominator * nominator / denominator;
          if (z > max_z[2])
          {
            max_z[2] = z;
            best_z_index[2] = j;
            lambda[2] = nominator / (denominator / 2);
          }
        }
      }
    }
  }

  // choose the best type
  absolute_best_z = -1;
  for (i = 0; i < 3; i++)
  {
    if ((type_selected[i] == 1) && (max_z[i] > absolute_best_z))
    {
      absolute_best_z = max_z[i];
      working_set_type = (char)i;
      work[0] = selected_indices[i][0];
      work[1] = best_z_index[i];
      if (i == 0)
      {
        work[0] += l;
        work[1] += l;
      }
      if (i == 2)
        work[2] = selected_indices[i][2] + l;
      lambda_star = lambda[i];
    }
  }

  active[work[0]] = true;
  active[work[1]] = true;
  if (working_set_type == 2)
  {
    active[work[2]] = true;
    working_set_size = 3;
  }
  else
    working_set_size = 2;

  if (absolute_best_z == -1)
  {
    working_set_size = -1;
    return 1;
  }

  switch (working_set_type)
  {
  case BETAS:
    best_u[0] = -1;
    best_u[1] = 1;
    lambda_star = min(lambda_star, beta[work[0] - l]);
    break;
  case ALPHAS:
    best_u[0] = -1;
    best_u[1] = 1;
    lambda_star = min(lambda_star, alpha[work[0]]);
    break;
  case ALPHAS_BETAS:
    if (lambda_star > 0)
    {
      best_u[0] = -1;
      best_u[1] = -1;
      best_u[2] = 2;
      lambda_star = min(lambda_star, min(alpha[work[0]], alpha[work[1]]));
    }
    else
    {
      lambda_star = -lambda_star;
      best_u[0] = 1;
      best_u[1] = 1;
      best_u[2] = -2;
      lambda_star = min(lambda_star, beta[work[2] - l] / 2);
    }
  }

  return 0;
}

void Solver_plus::reconstruct_gradient_plus()
{
  int i, j, true_i, act_set_i;

  if (active_size < l)
  {
    for (i = active_size; i < l; i++)
    {
      const Qfloat *Q_i = Q->get_Q(i, l);
      const Qfloat *Q_i_star = Q_star->get_Q(i, l);

      true_i = active_set[i];
      act_set_i = true_act_set_beta[true_i];

      const Qfloat *Q_i_star_beta = Q_star_beta->get_Q(act_set_i, l);
      G[i] = 0;
      g[i] = g_init[i];
      for (j = 0; j < l; j++)
        if (alpha[j] > 1e-8)
        {
          G[i] += alpha[j] * y[j] * Q_i[j];
          g[i] += alpha[j] * Q_i_star[j];
        }
      for (j = 0; j < l; j++)
        if (beta[j] > 1e-8)
          g[i] += beta[j] * Q_i_star_beta[j];
    }
  }

  if (active_size_beta < l)
  {
    for (i = active_size_beta; i < l; i++)
    {
      const Qfloat *Q_i_star_beta = Q_star_beta->get_Q(i, l);

      true_i = active_set_beta[i];
      act_set_i = true_act_set[true_i];
      const Qfloat *Q_i_star = Q_star->get_Q(act_set_i, l);

      g_beta[i] = g_beta_init[i];

      for (j = 0; j < l; j++)
        if (beta[j] > 1e-8)
          g_beta[i] += beta[j] * Q_i_star_beta[j];

      for (j = 0; j < l; j++)
        if (alpha[j] > 1e-8)
          g_beta[i] += alpha[j] * Q_i_star[j];
    }
  }
}

int Solver_plus::select_working_set(double *u, double &lambda_star)
{
  static double gain_hmg, gain2, tmp_lambda;
  static int tmp_depth, tmp_working_set_size, i, result;
  static char tmp_working_set_type;

  if (select_working_set_plus_hmg(u, lambda_star, gain_hmg) != 0)
  {
    // cannot find conjugate direction, try to go in the  direction of gradient
    memcpy(tmp_work, work, sizeof_int * working_set_size);
    tmp_depth = curr_depth;
    tmp_working_set_size = working_set_size;
    tmp_working_set_type = working_set_type;
    for (i = 0; i < working_set_size; i++)
    {
      active[work[i]] = false;
      work[i] = -1;
    }
    curr_depth = 0;
    working_set_size = 0;
    result = select_working_set_plus_hmg2(u, lambda_star, gain2);
    if (result == 1)
    {
      curr_depth = tmp_depth;
      working_set_size = tmp_working_set_size;
      working_set_type = tmp_working_set_type;
      memcpy(work, tmp_work, sizeof_int * tmp_working_set_size);
      for (i = 0; i < working_set_size; i++)
        active[work[i]] = true;
    }
    return result;
  }
  else
  {
    // store current direction, step size and the working set
    memcpy(tmp_work, work, sizeof_int * working_set_size);
    memcpy(tmp_u, u, sizeof_double * working_set_size);
    tmp_depth = curr_depth;
    tmp_working_set_size = working_set_size;
    tmp_working_set_type = working_set_type;
    tmp_lambda = lambda_star;
    for (i = 0; i < working_set_size; i++)
    {
      active[work[i]] = false;
      work[i] = -1;
    }
    curr_depth = 0;
    working_set_size = 0;
    if (select_working_set_plus_hmg2(u, lambda_star, gain2) == 0)
    {
      if (gain_hmg > gain2)
      {
        for (i = 0; i < working_set_size; i++)
          active[work[i]] = false;
        curr_depth = tmp_depth;
        working_set_size = tmp_working_set_size;
        working_set_type = tmp_working_set_type;
        lambda_star = tmp_lambda;
        memcpy(work, tmp_work, sizeof_int * tmp_working_set_size);
        memcpy(u, tmp_u, sizeof_double * working_set_size);
        for (i = 0; i < working_set_size; i++)
          active[work[i]] = true;
      }
    }
    else
    {
      curr_depth = tmp_depth;
      working_set_size = tmp_working_set_size;
      working_set_type = tmp_working_set_type;
      lambda_star = tmp_lambda;
      memcpy(u, tmp_u, sizeof_double * working_set_size);
      memcpy(work, tmp_work, sizeof_int * tmp_working_set_size);
      for (i = 0; i < working_set_size; i++)
        active[work[i]] = true;
    }
    return 0;
  }
}

void Solver_plus::Solve_plus_cg(int l, const QMatrix &Q, const QMatrix &Q_star, const QMatrix &Q_star_beta, const schar *y_,
                                double *alpha_, double *beta_, double Cp, double Cn, double tau_, double eps,
                                Solver::SolutionInfo *si, int shrinking)
{
  //debug lines added by Okba BEKHELIFI
  info("Running Solve_plus_cg\n");
  int i, j;

  this->l = l;
  this->Q = &Q;
  this->Q_star = &Q_star;
  this->Q_star_beta = &Q_star_beta;
  QD = Q.get_QD();
  QD_star = Q_star.get_QD();
  QD_star_beta = Q_star_beta.get_QD();
  clone(alpha, alpha_, l);
  clone(beta, beta_, l);
  this->Cp = Cp;
  this->Cn = Cn;
  this->eps = eps;
  tau = tau_;
  unshrink = false;
  prev_depth = -1;
  curr_depth = 0;
  working_set_size = 0;

  int l2 = 2 * l;
  double *G, *g, *g_beta;
  bool done_shrinking = false;

  y = new schar[l2];
  memcpy(y, y_, sizeof_char * l);
  for (i = l; i < l2; i++)
    y[i] = 0;

  work = new int[max_depth + 3];
  for (i = 0; i < max_depth + 3; i++)
    work[i] = -1;

  active = new bool[l2];
  for (i = 0; i < l2; i++)
    active[i] = false;

  // initialize alpha's
  alpha_cg = new double *[max_depth + 1];
  for (i = 0; i < max_depth + 1; i++)
    clone(alpha_cg[i], alpha_, l);

  // initialize beta's
  beta_cg = new double *[max_depth + 1];
  for (i = 0; i < max_depth + 1; i++)
    clone(beta_cg[i], beta_, l);

  // initialize alpha_status
  alpha_status_cg = new char *[max_depth + 1];
  for (i = 0; i < max_depth + 1; i++)
  {
    alpha_status_cg[i] = new char[l];
    for (j = 0; j < l; j++)
    {
      update_alpha_status_cg(j, i);
    }
  }

  // initialize beta_status
  beta_status_cg = new char *[max_depth + 1];
  for (i = 0; i < max_depth + 1; i++)
  {
    beta_status_cg[i] = new char[l];
    for (j = 0; j < l; j++)
    {
      update_beta_status_cg(j, i);
    }
  }

  // initialize gradient
  G_cg = new double *[max_depth + 1];
  g_cg = new double *[max_depth + 1];
  g_beta_cg = new double *[max_depth + 1];

  G_cg[0] = new double[l];
  g_cg[0] = new double[l];
  g_beta_cg[0] = new double[l];
  g_init = new double[l];
  g_beta_init = new double[l];

  G = G_cg[0];
  g = g_cg[0];
  g_beta = g_beta_cg[0];

  for (i = 0; i < l; i++)
  {
    G[i] = 0;
    g[i] = 0;
    g_init[i] = 0;
  }

  for (i = 0; i < l; i++)
  {
    const Qfloat *Q_i_star = Q_star.get_Q(i, l);

    for (j = 0; j < l; j++)
    {
      g[j] -= Cp * Q_i_star[j];
      g_init[j] -= Cp * Q_i_star[j];
    }

    if (!is_lower_bound_cg(i, 0))
    {
      const Qfloat *Q_i = Q.get_Q(i, l);
      double alpha_i = alpha[i];
      double y_i = y[i];
      for (j = 0; j < l; j++)
      {
        G[j] += alpha_i * y_i * Q_i[j];
        g[j] += alpha_i * Q_i_star[j];
      }
    }

    if (!is_lower_bound_beta_cg(i, 0))
    {
      double beta_i = beta[i];
      for (j = 0; j < l; j++)
        g[j] += beta_i * Q_i_star[j];
    }
  }

  clone(g_beta, g, l);
  clone(g_beta_init, g_init, l);
  for (i = 1; i < max_depth + 1; i++)
  {
    clone(G_cg[i], G, l);
    clone(g_cg[i], g, l);
    clone(g_beta_cg[i], g_beta, l);
  }

  active_set = new int[l];
  active_set_beta = new int[l];
  true_act_set = new int[l];
  true_act_set_beta = new int[l];
  for (i = 0; i < l; i++)
  {
    active_set[i] = i;
    active_set_beta[i] = i;
    true_act_set[i] = i;
    true_act_set_beta[i] = i;
  }
  active_size = l;
  active_size_beta = l;

  int iter = 0, counter = min(l, 1000) + 1, next_depth, n_conjugate = 0, worki, work0;
  bool corner = false;
  double *tmp_alpha, *tmp_beta, *tmp_G, *tmp_g, *tmp_g_beta;
  char *tmp_status, *tmp_status_beta;
  double *u, lambda_star, *alpha, *beta;
  Qfloat *Q_i_star_beta, *Q_i, *Q_i_star, *Q_k_star, *Q_k_star_beta;
  double *alpha_old, *beta_old, diff_i, diff_i_y, diff_k;
  int r, true_i, act_set_i, true_k, act_set_k;
  double gain2;

  tmp_work = new int[max_depth + 3];
  u = new double[max_depth + 3];
  tmp_u = new double[max_depth + 3];

  while (1)
  {

    if (--counter == 0)
    {
      counter = min(l, 1000);
      if (shrinking)
        done_shrinking = do_shrinking_plus_cg();
    }

    lambda_star = 0;

    if (corner)
    {
      if (wss_first_order(u, lambda_star) != 0)
      {
        reconstruct_gradient_plus_cg();
        active_size = l;
        active_size_beta = l;
        if (wss_first_order(u, lambda_star) != 0)
          break;
        else
          counter = 1;
      }
    }
    else if (curr_depth > 0)
    {
      if (select_working_set(u, lambda_star) != 0)
      {
        // reconstruct the whole gradient
        reconstruct_gradient_plus_cg();
        // reset active set size
        active_size = l;
        active_size_beta = l;
        if (select_working_set(u, lambda_star) == 0)
          counter = 1;
        else
          break;
      }
    }
    else
    {
      if (select_working_set_plus_hmg2(u, lambda_star, gain2) != 0)
      {
        reconstruct_gradient_plus_cg();
        active_size = l;
        active_size_beta = l;
        if (select_working_set_plus_hmg2(u, lambda_star, gain2) == 0)
          counter = 1;
        else
          break;
      }
    }

    iter++;
    // fprintf(stdout,"iter=%d\n",iter);
    // fflush(stdout);
    prev_depth = curr_depth;

    // shift old alpha's, G's and G_bar's
    n_conjugate++;
    next_depth = min(curr_depth + 1, max_depth);
    tmp_alpha = alpha_cg[next_depth];
    tmp_beta = beta_cg[next_depth];
    tmp_G = G_cg[next_depth];
    tmp_g = g_cg[next_depth];
    tmp_g_beta = g_beta_cg[next_depth];
    tmp_status = alpha_status_cg[next_depth];
    tmp_status_beta = beta_status_cg[next_depth];

    for (i = next_depth; i > 0; i--)
    {
      alpha_cg[i] = alpha_cg[i - 1];
      beta_cg[i] = beta_cg[i - 1];
      G_cg[i] = G_cg[i - 1];
      g_cg[i] = g_cg[i - 1];
      g_beta_cg[i] = g_beta_cg[i - 1];
      alpha_status_cg[i] = alpha_status_cg[i - 1];
      beta_status_cg[i] = beta_status_cg[i - 1];
    }

    if (!done_shrinking || (curr_depth == max_depth && curr_depth == prev_depth))
    {
      memcpy(tmp_alpha, alpha_cg[0], active_size * sizeof_double);
      memcpy(tmp_beta, beta_cg[0], active_size_beta * sizeof_double);
      memcpy(tmp_G, G_cg[0], active_size * sizeof_double);
      memcpy(tmp_g, g_cg[0], active_size * sizeof_double);
      memcpy(tmp_g_beta, g_beta_cg[0], active_size_beta * sizeof_double);
      memcpy(tmp_status, alpha_status_cg[0], active_size * sizeof_char);
      memcpy(tmp_status_beta, beta_status_cg[0], active_size_beta * sizeof_char);
      done_shrinking = false;
    }
    else
    {
      memcpy(tmp_alpha, alpha_cg[0], l * sizeof_double);
      memcpy(tmp_beta, beta_cg[0], l * sizeof_double);
      memcpy(tmp_G, G_cg[0], l * sizeof_double);
      memcpy(tmp_g, g_cg[0], l * sizeof_double);
      memcpy(tmp_g_beta, g_beta_cg[0], l * sizeof_double);
      memcpy(tmp_status, alpha_status_cg[0], l * sizeof_char);
      memcpy(tmp_status_beta, beta_status_cg[0], l * sizeof_char);
    }

    alpha_cg[0] = tmp_alpha;
    beta_cg[0] = tmp_beta;
    G_cg[0] = tmp_G;
    g_cg[0] = tmp_g;
    g_beta_cg[0] = tmp_g_beta;
    alpha_status_cg[0] = tmp_status;
    beta_status_cg[0] = tmp_status_beta;
    curr_depth = next_depth;

    alpha = alpha_cg[0];
    beta = beta_cg[0];
    g_beta = g_beta_cg[0];
    g = g_cg[0];
    G = G_cg[0];
    alpha_old = alpha_cg[1];
    beta_old = beta_cg[1];

    for (i = 0; i < working_set_size; i++)
    {
      worki = work[i];
      if (worki < l)
      {
        alpha[worki] += lambda_star * u[i];
        update_alpha_status_cg(worki, 0);
      }
      else
      {
        beta[worki - l] += lambda_star * u[i];
        update_beta_status_cg(worki - l, 0);
      }
    }
    // fflush(stdout);

    // check if we are at the corner
    corner = true;
    for (i = 0; i < working_set_size; i++)
    {
      worki = work[i];
      if (worki < l)
      {
        if (alpha[worki] >= 1e-8)
        {
          corner = false;
          break;
        }
      }
      else if (beta[worki - l] >= 1e-8)
      {
        corner = false;
        break;
      }
    }

    // update gradients
    switch (working_set_type)
    {
    case BETAS:
      for (i = 0; i < working_set_size; i++)
      {
        work0 = work[i] - l;
        Q_i_star_beta = Q_star_beta.get_Q(work0, active_size_beta);
        diff_i = beta[work0] - beta_old[work0];
        for (r = 0; r < active_size_beta; r++)
          g_beta[r] += diff_i * Q_i_star_beta[r];

        true_i = active_set_beta[work0];
        act_set_i = true_act_set[true_i];
        Q_i_star = Q_star.get_Q(act_set_i, active_size);

        for (r = 0; r < active_size; r++)
          g[r] += diff_i * Q_i_star[r];
      }
      break;

    case ALPHAS:
    case ALPHAS_DIFF_SIGN:
      for (i = 0; i < working_set_size; i++)
      {
        work0 = work[i];
        Q_i = Q.get_Q(work0, active_size);
        Q_i_star = Q_star.get_Q(work0, active_size);
        diff_i = alpha[work0] - alpha_old[work0];
        diff_i_y = diff_i * y[work0];

        for (r = 0; r < active_size; r++)
        {
          G[r] += diff_i_y * Q_i[r];
          g[r] += diff_i * Q_i_star[r];
        }

        true_i = active_set[work0];
        act_set_i = true_act_set_beta[true_i];
        Q_i_star_beta = Q_star_beta.get_Q(act_set_i, active_size_beta);

        for (r = 0; r < active_size_beta; r++)
          g_beta[r] += diff_i * Q_i_star_beta[r];
      }
      break;

    case ALPHAS_BETAS:
      for (i = 0; i < working_set_size; i++)
        if (work[i] < l)
        {
          work0 = work[i];
          Q_i = Q.get_Q(work0, active_size);
          Q_i_star = Q_star.get_Q(work0, active_size);
          diff_i = alpha[work0] - alpha_old[work0];
          diff_i_y = diff_i * y[work0];

          for (r = 0; r < active_size; r++)
          {
            G[r] += diff_i_y * Q_i[r];
            g[r] += diff_i * Q_i_star[r];
          }

          true_i = active_set[work0];
          act_set_i = true_act_set_beta[true_i];
          Q_i_star_beta = Q_star_beta.get_Q(act_set_i, active_size_beta);

          for (r = 0; r < active_size_beta; r++)
            g_beta[r] += diff_i * Q_i_star_beta[r];

          update_alpha_status_cg(work0, 0);
        }
        else
        {
          work0 = work[i] - l;
          Q_k_star_beta = Q_star_beta.get_Q(work0, active_size_beta);
          true_k = active_set_beta[work0];
          act_set_k = true_act_set[true_k];
          Q_k_star = Q_star.get_Q(act_set_k, active_size);
          diff_k = beta[work0] - beta_old[work0];
          for (r = 0; r < active_size; r++)
            g[r] += diff_k * Q_k_star[r];
          for (r = 0; r < active_size_beta; r++)
            g_beta[r] += diff_k * Q_k_star_beta[r];
        }
    }
  }
  calculate_rho_plus_cg(si->rho, si->rho_star);
  
  // put back the solution
  alpha = alpha_cg[0];
  beta = beta_cg[0];
  for (i = 0; i < l; i++)
  {
    alpha_[active_set[i]] = alpha[i];
    beta_[active_set_beta[i]] = beta[i];
  }

   // calculate objective value
  /*
  double v = 0;
  for (i = 0; i < l; i++)
    v += alpha_cg[0][i] * (G_cg[0][i] + p[i]);

  si->obj = v / 2;
  */
  //info("Objective value= %f\n", si->obj);

  info("\noptimization finished, #iter = %d\n", iter);

  si->rho *= -1;

  //
  
  for(i=0; i<max_depth+1; i++) {
    delete[] alpha_cg[i];
    delete[] beta_cg[i];
    delete[] alpha_status_cg[i];
    delete[] beta_status_cg[i];
    delete[] G_cg[i];
    delete[] g_cg[i];
    delete[] g_beta_cg[i];
  }

  delete[] alpha_cg;
  delete[] beta_cg;
  delete[] alpha_status_cg;
  delete[] beta_status_cg;
  delete[] G_cg;
  delete[] g_cg;
  delete[] g_beta_cg;
  delete[] g_init;
  delete[] g_beta_init;
  delete[] active;
  delete[] y;
  delete[] work;
  delete[] u;
  delete[] tmp_work;
  delete[] tmp_u;

}

void Solver_plus::Solve_plus(int l, const QMatrix &Q, const QMatrix &Q_star, const QMatrix &Q_star_beta, const schar *y_,
                             double *alpha_, double *beta_, double Cp, double Cn, double tau_, double eps,
                             Solver::SolutionInfo *si, int shrinking)
{
  //debug lines added by Okba BEKHELIFI
  // info("Running Solve_plus\n");

  int i, j;

  // Replacement for windows
  /*
  struct tms init_time, fin_time;
  long t_ini, t_fin;
  double  net_time;
  */
  clock_t begin, end;
  //
  this->l = l;
  this->Q = &Q;
  this->Q_star = &Q_star;
  this->Q_star_beta = &Q_star_beta;
  QD = Q.get_QD();
  QD_star = Q_star.get_QD();
  QD_star_beta = Q_star_beta.get_QD();
  clone(y, y_, l);
  clone(alpha, alpha_, l);
  clone(beta, beta_, l);
  this->Cp = Cp;
  this->Cn = Cn;
  this->eps = eps;
  tau = tau_;
  unshrink = false;

  alpha_status = new char[l];
  for (i = 0; i < l; i++)
    update_alpha_status(i);
  beta_status = new char[l];
  for (i = 0; i < l; i++)
    update_beta_status(i);

  //t_ini = times( &init_time);
  begin = clock();
  // initialize gradient
  G = new double[l];
  g = new double[l];
  g_beta = new double[l];
  g_init = new double[l];
  g_beta_init = new double[l];

  for (i = 0; i < l; i++)
  {
    G[i] = 0;
    g[i] = 0;
    g_init[i] = 0;
  }

  for (i = 0; i < l; i++)
  {
    const Qfloat *Q_i_star = Q_star.get_Q(i, l);

    for (j = 0; j < l; j++)
    {
      g[j] -= Cp * Q_i_star[j];
      g_init[j] -= Cp * Q_i_star[j];
    }

    if (!is_lower_bound(i))
    {
      const Qfloat *Q_i = Q.get_Q(i, l);
      double alpha_i = alpha[i];
      double y_i = y[i];
      for (j = 0; j < l; j++)
      {
        G[j] += alpha_i * y_i * Q_i[j];
        g[j] += alpha_i * Q_i_star[j];
      }
    }

    if (!is_lower_bound_beta(i))
    {
      double beta_i = beta[i];
      for (j = 0; j < l; j++)
        g[j] += beta_i * Q_i_star[j];
    }
  }
  for (i = 0; i < l; i++)
  {
    g_beta[i] = g[i];
    g_beta_init[i] = g_init[i];
  }

  active_set = new int[l];
  active_set_beta = new int[l];
  true_act_set = new int[l];
  true_act_set_beta = new int[l];
  for (int i = 0; i < l; i++)
  {
    active_set[i] = i;
    active_set_beta[i] = i;
    true_act_set[i] = i;
    true_act_set_beta[i] = i;
  }
  active_size = l;
  active_size_beta = l;

  int counter = min(l, 1000) + 1;

  // optimization step
  int iter = 0, y_i, y_j;
  Qfloat *Q_i, *Q_j, *Q_i_star, *Q_j_star, *Q_k_star, *Q_i_star_beta, *Q_j_star_beta, *Q_k_star_beta;
  double Delta, beta_i_old, beta_j_old, alpha_i_old, alpha_j_old, beta_k_old, nominator, denominator, min_alpha, alpha_change;
  double diff_i, diff_j, diff_k, beta_i, beta_k, alpha_i, diff_i_y, diff_j_y;
  int true_i, true_j, true_k, act_set_i, act_set_j, act_set_k;

  while (iter < 1e7)
  {

    int i, j, k, set_type, r;

    if (--counter == 0)
    {
      counter = min(l, 1000);
      if (shrinking)
        do_shrinking_plus();
    }

    if (select_working_set_plus(set_type, i, j, k, iter) != 0)
    {

      // reconstruct the whole gradient
      reconstruct_gradient_plus();

      // reset active set size and check
      active_size = l;
      active_size_beta = l;

      if (select_working_set_plus(set_type, i, j, k, iter) != 0)
        break;
      else
        counter = 1; // do shrinking next iteration
    }

    ++iter;

    switch (set_type)
    {

    case BETA_I_BETA_J:
      Q_i_star_beta = Q_star_beta.get_Q(i, active_size_beta);
      Q_j_star_beta = Q_star_beta.get_Q(j, active_size_beta);
      beta_i_old = beta[i];
      beta_j_old = beta[j];
      Delta = beta_i_old + beta_j_old;
      beta[i] += (g_beta[j] - g_beta[i]) / (Q_i_star_beta[i] + Q_j_star_beta[j] - 2 * Q_i_star_beta[j]);
      beta_i = beta[i];
      if (beta_i < 0)
        beta[i] = 0;
      if (beta_i > Delta)
        beta[i] = Delta;
      beta[j] = Delta - beta[i];

      diff_i = beta[i] - beta_i_old;
      diff_j = beta[j] - beta_j_old;
      for (r = 0; r < active_size_beta; r++)
        g_beta[r] += diff_i * Q_i_star_beta[r] + diff_j * Q_j_star_beta[r];

      true_i = active_set_beta[i];
      act_set_i = true_act_set[true_i];
      true_j = active_set_beta[j];
      act_set_j = true_act_set[true_j];
      Q_i_star = Q_star.get_Q(act_set_i, active_size);
      Q_j_star = Q_star.get_Q(act_set_j, active_size);

      for (r = 0; r < active_size; r++)
        g[r] += diff_i * Q_i_star[r] + diff_j * Q_j_star[r];

      update_beta_status(i);
      update_beta_status(j);
      // fprintf(stdout,"beta_i_old=%f beta_i=%f beta_j_old=%f beta_j=%f\n",beta_i_old,beta[i],beta_j_old,beta[j]);
      // fflush(stdout);
      break;

    case ALPHA_I_ALPHA_J:
      Q_i = Q.get_Q(i, active_size);
      Q_j = Q.get_Q(j, active_size);
      Q_i_star = Q_star.get_Q(i, active_size);
      Q_j_star = Q_star.get_Q(j, active_size);
      alpha_i_old = alpha[i];
      alpha_j_old = alpha[j];
      y_i = y[i];
      y_j = y[j];
      Delta = alpha_i_old + alpha_j_old;
      nominator = y_j * G[j] - y_i * G[i] + (g[j] - g[i]) / tau;
      denominator = Q_i[i] + Q_j[j] - 2 * Q_i[j] + (Q_i_star[i] + Q_j_star[j] - 2 * Q_i_star[j]) / tau;
      alpha[i] += nominator / denominator;
      alpha_i = alpha[i];
      if (alpha_i < 0)
        alpha[i] = 0;
      if (alpha_i > Delta)
        alpha[i] = Delta;
      alpha[j] = Delta - alpha[i];

      diff_i = alpha[i] - alpha_i_old;
      diff_j = alpha[j] - alpha_j_old;
      diff_i_y = diff_i * y_i;
      diff_j_y = diff_j * y_j;
      for (r = 0; r < active_size; r++)
      {
        G[r] += diff_i_y * Q_i[r] + diff_j_y * Q_j[r];
        g[r] += diff_i * Q_i_star[r] + diff_j * Q_j_star[r];
      }

      true_i = active_set[i];
      act_set_i = true_act_set_beta[true_i];
      true_j = active_set[j];
      act_set_j = true_act_set_beta[true_j];
      Q_i_star_beta = Q_star_beta.get_Q(act_set_i, active_size_beta);
      Q_j_star_beta = Q_star_beta.get_Q(act_set_j, active_size_beta);

      for (r = 0; r < active_size_beta; r++)
        g_beta[r] += diff_i * Q_i_star_beta[r] + diff_j * Q_j_star_beta[r];

      update_alpha_status(i);
      update_alpha_status(j);
      break;

    case ALPHA_I_ALPHA_J_BETA_K:
      Q_i = Q.get_Q(i, active_size);
      Q_j = Q.get_Q(j, active_size);
      Q_i_star = Q_star.get_Q(i, active_size);
      Q_j_star = Q_star.get_Q(j, active_size);
      Q_k_star_beta = Q_star_beta.get_Q(k, active_size_beta);

      true_k = active_set_beta[k];
      act_set_k = true_act_set[true_k];
      Q_k_star = Q_star.get_Q(act_set_k, active_size);

      alpha_i_old = alpha[i];
      alpha_j_old = alpha[j];
      beta_k_old = beta[k];
      y_i = y[i];
      y_j = y[j];
      if (alpha_i_old < alpha_j_old)
        min_alpha = alpha_i_old;
      else
        min_alpha = alpha_j_old;
      Delta = beta_k_old + 2 * min_alpha;
      nominator = y[i] * G[i] + y[j] * G[j] - 2 + (g[i] + g[j] - 2 * g_beta[k]) / tau;
      denominator = Q_i[i] + Q_j[j] - 2 * Q_i[j] + (Q_i_star[i] + Q_j_star[j] + 2 * Q_i_star[j] - 4 * Q_k_star[i] - 4 * Q_k_star[j] + 4 * Q_k_star_beta[k]) / tau;
      beta[k] += 2 * nominator / denominator;
      beta_k = beta[k];
      if (beta_k < 0)
        beta[k] = 0;
      if (beta_k > Delta)
        beta[k] = Delta;
      alpha_change = (beta_k_old - beta[k]) / 2;
      alpha[i] += alpha_change;
      alpha[j] += alpha_change;

      diff_i = alpha[i] - alpha_i_old;
      diff_j = alpha[j] - alpha_j_old;
      diff_k = beta[k] - beta_k_old;
      diff_i_y = diff_i * y_i;
      diff_j_y = diff_j * y_j;

      for (r = 0; r < active_size; r++)
      {
        G[r] += diff_i_y * Q_i[r] + diff_j_y * Q_j[r];
        g[r] += diff_i * Q_i_star[r] + diff_j * Q_j_star[r] + diff_k * Q_k_star[r];
      }

      true_i = active_set[i];
      act_set_i = true_act_set_beta[true_i];
      true_j = active_set[j];
      act_set_j = true_act_set_beta[true_j];
      Q_i_star_beta = Q_star_beta.get_Q(act_set_i, active_size_beta);
      Q_j_star_beta = Q_star_beta.get_Q(act_set_j, active_size_beta);

      for (r = 0; r < active_size_beta; r++)
        g_beta[r] += diff_i * Q_i_star_beta[r] + diff_j * Q_j_star_beta[r] + diff_k * Q_k_star_beta[r];

      update_alpha_status(i);
      update_alpha_status(j);
      update_beta_status(k);

      break;
    }
  }

  calculate_rho_plus(si->rho, si->rho_star);

  // No calculation of the objective value

  // time
  // t_fin = times( &fin_time);  
  // net_time = (double) (fin_time.tms_utime - init_time.tms_utime)/HZ;
  end = clock();
  double diffticks = begin - end;
  double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
  info("Solver time = %3.3e\n", diffms);
  // fprintf(stdout, "Solver time = %3.3e\n", diffms);
  // fprintf(stdout,"Solver time = %3.3e\n", net_time);
  // fflush(stdout);
  

  // put back the solution
  for (i = 0; i < l; i++)
  {
    alpha_[active_set[i]] = alpha[i];
    beta_[active_set_beta[i]] = beta[i];
  }

  si->upper_bound_p = Cp;
  si->upper_bound_n = Cn;

  info("\noptimization finished, #iter = %d\n", iter);

  si->rho *= -1;

  delete[] G;
  delete[] g;
  delete[] g_init;
  delete[] g_beta;
  delete[] g_beta_init;
  delete[] alpha_status;
  delete[] beta_status;
  delete[] alpha;
  delete[] beta;
}

void Solver_plus::calculate_rho_plus(double &rho, double &rho_star)
{
  int i, pos_size = 0, neg_size = 0;
  double pos_sum = 0, neg_sum = 0;

  for (i = 0; i < active_size; i++)
    if (alpha[i] > 1e-8)
    {
      if (y[i] == 1)
      {
        pos_size++;
        pos_sum += 1 - G[i] - g[i] / tau;
      }
      else
      {
        neg_size++;
        neg_sum += -1 - G[i] + g[i] / tau;
      }
    }

  if (pos_size != 0)
    pos_sum /= pos_size;

  if (neg_size != 0)
    neg_sum /= neg_size;

  rho = (pos_sum + neg_sum) / 2;
  rho_star = pos_sum - rho;
}

void Solver_plus::calculate_rho_plus_cg(double &rho, double &rho_star)
{
  int i, pos_size = 0, neg_size = 0;
  double pos_sum = 0, neg_sum = 0;
  double *alpha, *G, *g;

  alpha = alpha_cg[0];
  G = G_cg[0];
  g = g_cg[0];

  for (i = 0; i < active_size; i++)
    if (alpha[i] > 1e-8)
    {
      if (y[i] == 1)
      {
        pos_size++;
        pos_sum += 1 - G[i] - g[i] / tau;
      }
      else
      {
        neg_size++;
        neg_sum += -1 - G[i] + g[i] / tau;
      }
    }

  if (pos_size != 0)
    pos_sum /= pos_size;

  if (neg_size != 0)
    neg_sum /= neg_size;

  rho = (pos_sum + neg_sum) / 2;
  rho_star = pos_sum - rho;
}

// return 1 if already optimal, return 0 otherwise
int Solver_plus::select_working_set_plus(int &set_type, int &i_out, int &j_out, int &k_out, int iter)
{
  double gap[3];

  for (int i = 0; i < 3; i++)
    gap[i] = -1;

  int i, j, best_B1 = -1, best_B1B2 = -1, best_A2 = -1, best_A2A4 = -1, best_A1 = -1, best_A1A3 = -1, i_ind, j_ind, k_ind;
  int type_selected[3], selected_indices[3][3];
  double max_B1 = -1e20, min_B1B2 = 1e20, max_A2 = -1e20, min_A2A4 = 1e20, max_A1 = -1e20, min_A1A3 = 1e20;
  double alpha_i, g_i, g_j, y_i, y_j, deriv_alpha_i, first_order_criterion;
  double max_z[3], z, absolute_best_z, nominator, nominator_base, denominator_base, j_deriv, tau2, Q_star_ii;
  int best_z_index[3], true_k, act_set_k;
  Qfloat *Q_i, *Q_i_star, *Q_k_star, *Q_k_star_beta;
  double deriv_alpha_i1, deriv_alpha_i2, nominator1, nominator2, nominator_base1, nominator_base2, j_deriv1, j_deriv2;
  double max_A2_1, max_A2_2, min_A2A4_1, min_A2A4_2, max_A1_1, max_A1_2, min_A1A3_1, min_A1A3_2;

  // first-order working set selection

  // compute all maxima and minima related to alphas
  for (i = 0; i < active_size; i++)
  {
    alpha_i = alpha[i];
    g_i = g[i];
    y_i = y[i];
    deriv_alpha_i1 = y_i * G[i];
    deriv_alpha_i2 = g_i;
    deriv_alpha_i = deriv_alpha_i1 + deriv_alpha_i2 / tau;

    // max A2
    if (alpha_i > 1e-8 && y_i == -1 && deriv_alpha_i > max_A2)
    {
      max_A2 = deriv_alpha_i;
      best_A2 = i;
      max_A2_1 = deriv_alpha_i1;
      max_A2_2 = deriv_alpha_i2;
    }

    // min A2A4
    if (y_i == -1 && deriv_alpha_i < min_A2A4)
    {
      min_A2A4 = deriv_alpha_i;
      best_A2A4 = i;
      min_A2A4_1 = deriv_alpha_i1;
      min_A2A4_2 = deriv_alpha_i2;
    }

    // max A1
    if (alpha_i > 1e-8 && y_i == 1 && deriv_alpha_i > max_A1)
    {
      max_A1 = deriv_alpha_i;
      best_A1 = i;
      max_A1_1 = deriv_alpha_i1;
      max_A1_2 = deriv_alpha_i2;
    }

    // min A1A3
    if (y_i == 1 && deriv_alpha_i < min_A1A3)
    {
      min_A1A3 = deriv_alpha_i;
      best_A1A3 = i;
      min_A1A3_1 = deriv_alpha_i1;
      min_A1A3_2 = deriv_alpha_i2;
    }
  }

  // compute all maxima and minima related to betas
  for (i = 0; i < active_size_beta; i++)
  {
    g_i = g_beta[i];

    // max B1
    if (beta[i] > 1e-8 && g_i > max_B1)
    {
      max_B1 = g_i;
      best_B1 = i;
    }

    // min B1B2
    if (g_i < min_B1B2)
    {
      min_B1B2 = g_i;
      best_B1B2 = i;
    }
  }

  max_B1 /= tau;
  min_B1B2 /= tau;

  // select maximal violating pairs
  if (max_B1 - min_B1B2 < eps)
    type_selected[0] = 0;
  else
  {
    type_selected[0] = 1;
    selected_indices[0][0] = best_B1;
    selected_indices[0][1] = best_B1B2;
    gap[0] = max_B1 - min_B1B2;
  }

  if (((max_A2 - min_A2A4 < eps) || ((max_A2_1 - min_A2A4_1 < eps) && (max_A2_2 - min_A2A4_2 < eps))) &&
      ((max_A1 - min_A1A3 < eps) || ((max_A1_1 - min_A1A3_1 < eps) && (max_A1_2 - min_A1A3_2 < eps))))
    type_selected[1] = 0;
  else
  {
    if ((max_A2 - min_A2A4 > max_A1 - min_A1A3) && ((max_A2_1 - min_A2A4_1 >= eps) || (max_A2_2 - min_A2A4_2 >= eps)))
    {
      type_selected[1] = 1;
      selected_indices[1][0] = best_A2;
      selected_indices[1][1] = best_A2A4;
    }
    else
    {
      if ((max_A2 - min_A2A4 <= max_A1 - min_A1A3) && ((max_A1_1 - min_A1A3_1 >= eps) || (max_A1_2 - min_A1A3_2 >= eps)))
      {
        type_selected[1] = 1;
        selected_indices[1][0] = best_A1;
        selected_indices[1][1] = best_A1A3;
      }
      else
        type_selected[1] = 0;
    }
  }

  if (((2 * max_B1 + 2 - min_A1A3 - min_A2A4 < eps) || ((2 - min_A1A3_1 - min_A2A4_1 < eps) && (2 * max_B1 * tau - min_A1A3_2 - min_A2A4_2 < eps))) &&
      ((max_A1 + max_A2 - 2 * min_B1B2 - 2 < eps) || ((max_A1_1 + max_A2_1 - 2 < eps) && (max_A1_2 + max_A2_2 - 2 * min_B1B2 * tau < eps))))
    type_selected[2] = 0;
  else
  {
    if ((2 * max_B1 + 2 - min_A1A3 - min_A2A4 > max_A1 + max_A2 - 2 * min_B1B2 - 2) && ((2 - min_A1A3_1 - min_A2A4_1 >= eps) || (2 * max_B1 * tau - min_A1A3_2 - min_A2A4_2 >= eps)))
    {
      type_selected[2] = 1;
      selected_indices[2][0] = best_A1A3;
      selected_indices[2][1] = best_A2A4;
      selected_indices[2][2] = best_B1;
    }
    else
    {
      if ((2 * max_B1 + 2 - min_A1A3 - min_A2A4 <= max_A1 + max_A2 - 2 * min_B1B2 - 2) && ((max_A1_1 + max_A2_1 - 2 >= eps) || (max_A1_2 + max_A2_2 - 2 * min_B1B2 * tau >= eps)))
      {
        type_selected[2] = 1;
        selected_indices[2][0] = best_A1;
        selected_indices[2][1] = best_A2;
        selected_indices[2][2] = best_B1B2;
      }
      else
        type_selected[2] = 0;
    }
  }

  if (type_selected[0] + type_selected[1] + type_selected[2] == 0)
    return 1;

  for (i = 0; i < 3; i++)
    max_z[i] = -1e20;

  // second-order working set selection
  if (type_selected[0] == 1)
  {
    i_ind = selected_indices[0][0];
    g_i = g_beta[i_ind];
    Q_i_star = Q_star_beta->get_Q(i_ind, active_size_beta);
    Q_star_ii = Q_i_star[i_ind];
    tau2 = 2 * tau;
    for (j = 0; j < active_size_beta; j++)
    {
      g_j = g_beta[j];
      if (eps + g_j / tau < g_i / tau)
      {
        nominator = g_i - g_j;
        z = nominator * nominator / (tau2 * (Q_star_ii + QD_star_beta[j] - 2 * Q_i_star[j]));
        if (z > max_z[0])
        {
          max_z[0] = z;
          best_z_index[0] = j;
        }
      }
    }
  }

  if (type_selected[1] == 1)
  {
    i_ind = selected_indices[1][0];
    y_i = y[i_ind];
    Q_i = Q->get_Q(i_ind, active_size);
    Q_i_star = Q_star->get_Q(i_ind, active_size);
    nominator_base = y_i * G[i_ind] + g[i_ind] / tau;
    nominator_base1 = y_i * G[i_ind];
    nominator_base2 = g[i_ind];
    denominator_base = 2 * (Q_i[i_ind] + Q_i_star[i_ind] / tau);

    for (j = 0; j < active_size; j++)
    {
      y_j = y[j];
      j_deriv = y_j * G[j] + g[j] / tau;
      j_deriv1 = y_j * G[j];
      j_deriv2 = g[j];

      if (y_j == y_i && j_deriv + eps < nominator_base && ((j_deriv1 + eps < nominator_base1) || (j_deriv2 + eps < nominator_base2)))
      {
        nominator = nominator_base - j_deriv;
        z = nominator * nominator / (denominator_base + 2 * (QD[j] - 2 * Q_i[j] + (QD_star[j] - 2 * Q_i_star[j]) / tau));
        if (z > max_z[1])
        {
          max_z[1] = z;
          best_z_index[1] = j;
        }
      }
    }
  }

  if (type_selected[2] == 1)
  {
    i_ind = selected_indices[2][0];
    j_ind = selected_indices[2][1];
    k_ind = selected_indices[2][2];
    Q_i = Q->get_Q(i_ind, active_size);
    Q_i_star = Q_star->get_Q(i_ind, active_size);
    Q_k_star_beta = Q_star_beta->get_Q(k_ind, active_size_beta);

    true_k = active_set_beta[k_ind];
    act_set_k = true_act_set[true_k];
    Q_k_star = Q_star->get_Q(act_set_k, active_size);

    nominator_base = y[i_ind] * G[i_ind] - 2 + (g[i_ind] - 2 * g_beta[k_ind]) / tau;
    nominator_base1 = y[i_ind] * G[i_ind] - 2;
    nominator_base2 = g[i_ind] - 2 * g_beta[k_ind];
    denominator_base = 2 * (Q_i[i_ind] + (Q_i_star[i_ind] - 4 * Q_k_star[i_ind] + 4 * Q_k_star_beta[k_ind]) / tau);
    first_order_criterion = nominator_base + y[j_ind] * G[j_ind] + g[j_ind] / tau;
    for (j = 0; j < active_size; j++)
      if (y[j] == -1)
      {
        nominator1 = nominator_base1 + y[j] * G[j];
        nominator2 = nominator_base2 + g[j];
        nominator = nominator_base + y[j] * G[j] + g[j] / tau;
        if ((first_order_criterion < 0 && nominator < -eps && ((nominator1 < -eps) || (nominator2 < -eps))) ||
            (first_order_criterion > 0 && alpha[j] > 1e-8 && nominator > eps && ((nominator1 > eps) || (nominator2 > eps))))
        {
          z = nominator * nominator / (denominator_base + 2 * (QD[j] - 2 * Q_i[j] + (QD_star[j] + 2 * Q_i_star[j] - 4 * Q_k_star[j]) / tau));
          if (z > max_z[2])
          {
            max_z[2] = z;
            best_z_index[2] = j;
          }
        }
      }
  }

  // choose the best type
  absolute_best_z = -1;
  for (i = 0; i < 3; i++)
    if ((type_selected[i] == 1) && (max_z[i] > absolute_best_z))
    {
      absolute_best_z = max_z[i];
      set_type = i;
      i_out = selected_indices[i][0];
      j_out = best_z_index[i];
      if (i == 2)
        k_out = selected_indices[i][2];
    }

  return 0;
}

bool Solver_plus::be_shrunk_alpha(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4)
{
  int y_i = y[i];
  double deriv_i = y_i * G[i] + g[i] / tau;

  if (alpha[i] <= 1e-8)
  {
    if (y_i == 1 && deriv_i <= max_A1 + eps)
      return false;
    if (y_i == -1 && deriv_i <= max_A2 + eps)
      return false;
    return deriv_i + min_A1A3 + eps > 2 * max_B1 + 2;
  }
  else
  {
    if (y_i == 1)
      return max_A1 - deriv_i < eps && deriv_i - min_A1A3 < eps && 2 * max_B1 + 2 - deriv_i - min_A2A4 < eps && deriv_i + max_A2 - 2 * min_B1B2 - 2 < eps;
    else
      return max_A2 - deriv_i < eps && deriv_i - min_A2A4 < eps && 2 * max_B1 + 2 - min_A1A3 - deriv_i < eps && max_A1 + deriv_i - 2 * min_B1B2 - 2 < eps;
  }
}

bool Solver_plus::be_shrunk_beta(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4)
{
  double g_beta_i = g_beta[i] / tau;

  if (beta[i] <= 1e-8)
    return (g_beta_i + eps > max_B1 && 2 * g_beta_i + 2 + eps > max_A1 + max_A2);
  else
    return (g_beta_i - min_B1B2 < eps && max_B1 - g_beta_i < eps &&
            2 * g_beta_i + 2 - min_A1A3 - min_A2A4 < eps && max_A1 + max_A2 - 2 * g_beta_i - 2 < eps);
}

bool Solver_plus::be_shrunk_alpha_cg(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4)
{
  if (active[i])
    return false;

  int y_i = y[i];
  double deriv_i = y_i * G_cg[0][i] + g_cg[0][i] / tau;

  if (alpha_cg[0][i] <= 1e-8)
  {
    if (y_i == 1 && deriv_i <= max_A1 + eps)
      return false;
    if (y_i == -1 && deriv_i <= max_A2 + eps)
      return false;
    return deriv_i + min_A1A3 + eps > 2 * max_B1 + 2;
  }
  else
  {
    if (y_i == 1)
      return max_A1 - deriv_i < eps && deriv_i - min_A1A3 < eps && 2 * max_B1 + 2 - deriv_i - min_A2A4 < eps && deriv_i + max_A2 - 2 * min_B1B2 - 2 < eps;
    else
      return max_A2 - deriv_i < eps && deriv_i - min_A2A4 < eps && 2 * max_B1 + 2 - min_A1A3 - deriv_i < eps && max_A1 + deriv_i - 2 * min_B1B2 - 2 < eps;
  }
}

bool Solver_plus::be_shrunk_beta_cg(int i, double max_B1, double max_A1, double max_A2, double min_B1B2, double min_A1A3, double min_A2A4)
{
  if (active[i + l])
    return false;

  double g_beta_i = g_beta_cg[0][i] / tau;

  if (beta_cg[0][i] <= 1e-8)
    return (g_beta_i + eps > max_B1 && 2 * g_beta_i + 2 + eps > max_A1 + max_A2);
  else
    return (g_beta_i - min_B1B2 < eps && max_B1 - g_beta_i < eps &&
            2 * g_beta_i + 2 - min_A1A3 - min_A2A4 < eps && max_A1 + max_A2 - 2 * g_beta_i - 2 < eps);
}

void Solver_plus::do_shrinking_plus()
{
  int i, y_i;
  double g_i, alpha_i, deriv_alpha_i;
  double max_B1 = -1e20, min_B1B2 = 1e20, max_A2 = -1e20, min_A2A4 = 1e20, max_A1 = -1e20, min_A1A3 = 1e20;

  // compute all maxima and minima related to alphas
  for (i = 0; i < active_size; i++)
  {
    alpha_i = alpha[i];
    g_i = g[i];
    y_i = y[i];
    deriv_alpha_i = y_i * G[i] + g_i / tau;

    // max A2
    if (alpha_i > 1e-8 && y_i == -1 && deriv_alpha_i > max_A2)
      max_A2 = deriv_alpha_i;

    // min A2A4
    if (y_i == -1 && deriv_alpha_i < min_A2A4)
      min_A2A4 = deriv_alpha_i;

    // max A1
    if (alpha_i > 1e-8 && y_i == 1 && deriv_alpha_i > max_A1)
      max_A1 = deriv_alpha_i;

    // min A1A3max_A2, min_A2A4, max_A1, min_A1A3
    if (y_i == 1 && deriv_alpha_i < min_A1A3)
      min_A1A3 = deriv_alpha_i;
  }

  // compute all maxima and minima related to betas
  for (i = 0; i < active_size_beta; i++)
  {
    g_i = g_beta[i];

    // max B1
    if (beta[i] > 1e-8 && g_i > max_B1)
      max_B1 = g_i;

    // min B1B2
    if (g_i < min_B1B2)
      min_B1B2 = g_i;
  }

  if (unshrink == false && max_B1 - min_B1B2 < eps * 10 &&
      max_A2 - min_A2A4 < eps * 10 && max_A1 - min_A1A3 < eps * 10 &&
      2 * max_B1 + 2 - min_A1A3 - min_A2A4 < eps * 10 && max_A1 + max_A2 - 2 * min_B1B2 - 2 < eps * 10)
  {
    unshrink = true;
    reconstruct_gradient_plus();
    active_size = l;
    active_size_beta = l;
  }

  if (active_size_beta > 2)
  {
    for (i = 0; i < active_size_beta; i++)
    {
      if (active_size_beta <= 2)
        break;
      if (be_shrunk_beta(i, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
      {
        active_size_beta--;
        while (active_size_beta > i)
        {
          if (!be_shrunk_beta(active_size_beta, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
          {
            swap_index_beta(i, active_size_beta);
            break;
          }
          active_size_beta--;
          if (active_size_beta <= 2)
            break;
        }
      }
    }
  }

  for (i = 0; i < active_size; i++)
  {
    if (be_shrunk_alpha(i, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
    {
      active_size--;
      while (active_size > i)
      {
        if (!be_shrunk_alpha(active_size, max_B1, max_A1, max_A2, min_B1B2, min_A1A3, min_A2A4))
        {
          swap_index_alpha(i, active_size);
          break;
        }
        active_size--;
      }
    }
  }
}

// Kernel.h
//
// Q matrices for various formulations
//
class SVC_Q : public Kernel
{
public:
  SVC_Q(const svm_problem &prob, const svm_parameter &param, const schar *y_)
      : Kernel(prob.l, prob.x, param)
  {
    clone(y, y_, prob.l);
    cache = new Cache(prob.l, (long int)(param.cache_size * (1 << 20)));
    QD = new Qfloat[prob.l];
    for (int i = 0; i < prob.l; i++)
      QD[i] = (Qfloat)(this->*kernel_function)(i, i);
  }

  Qfloat *get_Q(int i, int len) const
  {
    Qfloat *data;
    int start, j;
    if ((start = cache->get_data(i, &data, len)) < len)
    {
      for (j = start; j < len; j++)
        data[j] = (Qfloat)(y[i] * y[j] * (this->*kernel_function)(i, j));
    }
    return data;
  }

  Qfloat *get_QD() const
  {
    return QD;
  }

  void swap_index(int i, int j) const
  {
    cache->swap_index(i, j);
    Kernel::swap_index(i, j);
    swap(y[i], y[j]);
    swap(QD[i], QD[j]);
  }

  ~SVC_Q()
  {
    delete[] y;
    delete cache;
    delete[] QD;
  }

private:
  schar *y;
  Cache *cache;
  Qfloat *QD;
};

class ONE_CLASS_Q : public Kernel
{
public:
  ONE_CLASS_Q(const svm_problem &prob, const svm_parameter &param)
      : Kernel(prob.l, prob.x, param)
  {
    cache = new Cache(prob.l, (long int)(param.cache_size * (1 << 20)));
    QD = new Qfloat[prob.l];
    for (int i = 0; i < prob.l; i++)
      QD[i] = (Qfloat)(this->*kernel_function)(i, i);
  }

  Qfloat *get_Q(int i, int len) const
  {
    Qfloat *data;
    int start, j;
    if ((start = cache->get_data(i, &data, len)) < len)
    {
      for (j = start; j < len; j++)
        data[j] = (Qfloat)(this->*kernel_function)(i, j);
    }
    return data;
  }

  Qfloat *get_QD() const
  {
    return QD;
  }

  void swap_index(int i, int j) const
  {
    cache->swap_index(i, j);
    Kernel::swap_index(i, j);
    swap(QD[i], QD[j]);
  }

  ~ONE_CLASS_Q()
  {
    delete cache;
    delete[] QD;
  }

private:
  Cache *cache;
  Qfloat *QD;
};

class SVR_Q : public Kernel
{
public:
  SVR_Q(const svm_problem &prob, const svm_parameter &param)
      : Kernel(prob.l, prob.x, param)
  {
    l = prob.l;
    cache = new Cache(l, (long int)(param.cache_size * (1 << 20)));
    QD = new Qfloat[2 * l];
    sign = new schar[2 * l];
    index = new int[2 * l];
    for (int k = 0; k < l; k++)
    {
      sign[k] = 1;
      sign[k + l] = -1;
      index[k] = k;
      index[k + l] = k;
      QD[k] = (Qfloat)(this->*kernel_function)(k, k);
      QD[k + l] = QD[k];
    }
    buffer[0] = new Qfloat[2 * l];
    buffer[1] = new Qfloat[2 * l];
    next_buffer = 0;
  }

  void swap_index(int i, int j) const
  {
    swap(sign[i], sign[j]);
    swap(index[i], index[j]);
    swap(QD[i], QD[j]);
  }

  Qfloat *get_Q(int i, int len) const
  {
    Qfloat *data;
    int j, real_i = index[i];
    if (cache->get_data(real_i, &data, l) < l)
    {
      for (j = 0; j < l; j++)
        data[j] = (Qfloat)(this->*kernel_function)(real_i, j);
    }

    // reorder and copy
    Qfloat *buf = buffer[next_buffer];
    next_buffer = 1 - next_buffer;
    schar si = sign[i];
    for (j = 0; j < len; j++)
      buf[j] = (Qfloat)si * (Qfloat)sign[j] * data[index[j]];
    return buf;
  }

  Qfloat *get_QD() const
  {
    return QD;
  }

  ~SVR_Q()
  {
    delete cache;
    delete[] sign;
    delete[] index;
    delete[] buffer[0];
    delete[] buffer[1];
    delete[] QD;
  }

private:
  int l;
  Cache *cache;
  schar *sign;
  int *index;
  mutable int next_buffer;
  Qfloat *buffer[2];
  Qfloat *QD;
};

// common.c
//
// construct and solve various formulations
//
static void solve_c_svc(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo *si, double Cp, double Cn)
{
  int l = prob->l;
  double *minus_ones = new double[l];
  schar *y = new schar[l];

  int i;

  for (i = 0; i < l; i++)
  {
    alpha[i] = 0;
    minus_ones[i] = -1;
    if (prob->y[i] > 0)
      y[i] = +1;
    else
      y[i] = -1;
  }

  Solver s(param->optimizer);
  if (param->optimizer != -1)
    s.Solve_cg(l, SVC_Q(*prob, *param, y), minus_ones, y,
               alpha, Cp, Cn, param->eps, si, param->shrinking);
  else
    s.Solve(l, SVC_Q(*prob, *param, y), minus_ones, y,
            alpha, Cp, Cn, param->eps, si, param->shrinking);

  double sum_alpha = 0;
  for (i = 0; i < l; i++)
    sum_alpha += alpha[i];

  if (Cp == Cn)
    info("nu = %f\n", sum_alpha / (Cp * prob->l));

  for (i = 0; i < l; i++)
    alpha[i] *= y[i];

  delete[] minus_ones;
  delete[] y;
}

static void solve_svm_plus(const svm_problem *prob, const svm_parameter *param,
                           double *alpha, double *beta, Solver::SolutionInfo *si, double Cp, double Cn)
{

 // info("Inside solve svm plus \n");

  int l = prob->l;
  schar *y = new schar[l];
  schar *y_true = new schar[l];
  svm_parameter cheat_param, cheat_param2;
  svm_problem cheat_problem, cheat_problem2;
  int i;
  int l_pos = 0, l_neg = 0;

  // initialize alphas and betas
  for (i = 0; i < l; i++)
  {
    alpha[i] = 0;
    beta[i] = Cp;
    y[i] = 1;
    if (prob->y[i] > 0)
    {
      y_true[i] = +1;
      l_pos++;
    }
    else
    {
      y_true[i] = -1;
      l_neg++;
    }
  }
  
  //info("init alphas and betas done successfully \n");

  cheat_param = *param;
  cheat_param.kernel_type = 2;
  cheat_param.gamma = param->gamma_star;
  cheat_problem = *prob;
  cheat_problem.x = Malloc(struct svm_node *, prob->l);
  memcpy(cheat_problem.x, prob->x_star, l * sizeof(struct svm_node *));
  //info("Memcopy cheat pb1 success\n");

  cheat_param2 = *param;
  cheat_param2.kernel_type = 2;
  cheat_param2.gamma = param->gamma_star;
  cheat_problem2 = *prob;
  cheat_problem2.x = Malloc(struct svm_node *, prob->l);
  memcpy(cheat_problem2.x, prob->x_star, l * sizeof(struct svm_node *));
  //info("Memcopy cheat pb2 success\n");

  /*
  info("Param:%d\n", *param);
  info("cheat Param:%d\n", cheat_param);
  info("cheat Param2:%d\n", cheat_param2);

  info("Prob:%d\n", *prob);
  info("cheat problem :%d\n", cheat_problem);
  info("cheat problem2:%d\n", cheat_problem2);
  */
  SVC_Q kernel1 = SVC_Q(*prob, *param, y);
  //info("Kernel 1 initialized correctly \n");
  SVC_Q kernel3 = SVC_Q(cheat_problem2, cheat_param2, y);
  //info("Kernel 3 initialized correctly \n");
  SVC_Q kernel2 = SVC_Q(cheat_problem, cheat_param, y);
  //info("Kernel 2 initialized correctly \n");

  //info("init kernels done\n");

  Solver_plus s(param->optimizer);

  //info("lets select optimizer\n");
  if (param->optimizer != -1)
  {
    //info("lets run solve plus cg\n");
    s.Solve_plus_cg(l, kernel1, kernel2,
                    kernel3, y_true,
                    alpha, beta, Cp, Cn, param->tau, param->eps, si, param->shrinking);
  }
  else
  {

    s.Solve_plus(l, kernel1, kernel2,
                 kernel3, y_true,
                 alpha, beta, Cp, Cn, param->tau, param->eps, si, param->shrinking);
  }
  // produce the same output as SVM
  for (i = 0; i < l; i++)
  {
    if (alpha[i] < 0)
      alpha[i] = 0;
    alpha[i] *= y_true[i];
  }

  delete[] y;
  delete[] y_true;
}

static void solve_nu_svc(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo *si)
{
  int i;
  int l = prob->l;
  double nu = param->nu;

  schar *y = new schar[l];

  for (i = 0; i < l; i++)
    if (prob->y[i] > 0)
      y[i] = +1;
    else
      y[i] = -1;

  double sum_pos = nu * l / 2;
  double sum_neg = nu * l / 2;

  for (i = 0; i < l; i++)
    if (y[i] == +1)
    {
      alpha[i] = min(1.0, sum_pos);
      sum_pos -= alpha[i];
    }
    else
    {
      alpha[i] = min(1.0, sum_neg);
      sum_neg -= alpha[i];
    }

  double *zeros = new double[l];

  for (i = 0; i < l; i++)
    zeros[i] = 0;

  Solver_NU s;
  s.Solve(l, SVC_Q(*prob, *param, y), zeros, y,
          alpha, 1.0, 1.0, param->eps, si, param->shrinking);
  double r = si->r;

  info("C = %f\n", 1 / r);

  for (i = 0; i < l; i++)
    alpha[i] *= y[i] / r;

  si->rho /= r;
  si->obj /= (r * r);
  si->upper_bound_p = 1 / r;
  si->upper_bound_n = 1 / r;

  delete[] y;
  delete[] zeros;
}

static void solve_one_class(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo *si)
{
  int l = prob->l;
  double *zeros = new double[l];
  schar *ones = new schar[l];
  int i;

  int n = (int)(param->nu * prob->l); // # of alpha's at upper bound

  for (i = 0; i < n; i++)
    alpha[i] = 1;
  if (n < prob->l)
    alpha[n] = param->nu * prob->l - n;
  for (i = n + 1; i < l; i++)
    alpha[i] = 0;

  for (i = 0; i < l; i++)
  {
    zeros[i] = 0;
    ones[i] = 1;
  }

  Solver s;
  s.Solve(l, ONE_CLASS_Q(*prob, *param), zeros, ones,
          alpha, 1.0, 1.0, param->eps, si, param->shrinking);

  delete[] zeros;
  delete[] ones;
}

static void solve_epsilon_svr(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo *si)
{
  int l = prob->l;
  double *alpha2 = new double[2 * l];
  double *linear_term = new double[2 * l];
  schar *y = new schar[2 * l];
  int i;

  for (i = 0; i < l; i++)
  {
    alpha2[i] = 0;
    linear_term[i] = param->p - prob->y[i];
    y[i] = 1;

    alpha2[i + l] = 0;
    linear_term[i + l] = param->p + prob->y[i];
    y[i + l] = -1;
  }

  Solver s;
  s.Solve(2 * l, SVR_Q(*prob, *param), linear_term, y,
          alpha2, param->C, param->C, param->eps, si, param->shrinking);

  double sum_alpha = 0;
  for (i = 0; i < l; i++)
  {
    alpha[i] = alpha2[i] - alpha2[i + l];
    sum_alpha += fabs(alpha[i]);
  }
  info("nu = %f\n", sum_alpha / (param->C * l));

  delete[] alpha2;
  delete[] linear_term;
  delete[] y;
}

static void solve_nu_svr(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo *si)
{
  int l = prob->l;
  double C = param->C;
  double *alpha2 = new double[2 * l];
  double *linear_term = new double[2 * l];
  schar *y = new schar[2 * l];
  int i;

  double sum = C * param->nu * l / 2;
  for (i = 0; i < l; i++)
  {
    alpha2[i] = alpha2[i + l] = min(sum, C);
    sum -= alpha2[i];

    linear_term[i] = -prob->y[i];
    y[i] = 1;

    linear_term[i + l] = prob->y[i];
    y[i + l] = -1;
  }

  Solver_NU s;
  s.Solve(2 * l, SVR_Q(*prob, *param), linear_term, y,
          alpha2, C, C, param->eps, si, param->shrinking);

  info("epsilon = %f\n", -si->r);

  for (i = 0; i < l; i++)
    alpha[i] = alpha2[i] - alpha2[i + l];

  delete[] alpha2;
  delete[] linear_term;
  delete[] y;
}

//
// decision_function
//
struct decision_function
{
  double *alpha;
  double *beta;
  double rho;
  double rho_star;
};

decision_function svm_train_one(const svm_problem *prob, const svm_parameter *param,
                                double Cp, double Cn)
{

  // info("Inside SVM TRAIN ONE\n");

  double *alpha = Malloc(double, prob->l);
  double *beta = NULL;

  if (param->svm_type == SVM_PLUS)
    beta = Malloc(double, prob->l);

  Solver::SolutionInfo si;
  switch (param->svm_type)
  {
  case C_SVC:
    // info("lets solve C_SVC\n");
    solve_c_svc(prob, param, alpha, &si, Cp, Cn);
    break;
  case SVM_PLUS:
    // info("lets solve svm plus\n");
    solve_svm_plus(prob, param, alpha, beta, &si, Cp, Cn);
    break;
  case NU_SVC:
    solve_nu_svc(prob, param, alpha, &si);
    break;
  case ONE_CLASS:
    solve_one_class(prob, param, alpha, &si);
    break;
  case EPSILON_SVR:
    solve_epsilon_svr(prob, param, alpha, &si);
    break;
  case NU_SVR:
    solve_nu_svr(prob, param, alpha, &si);
    break;
  }

  info("obj = %f, rho = %f\n", si.obj, si.rho);

  // output SVs
  int nSV = 0;
  int nSV_star = 0;
  int nBSV = 0;
  int nBSV_star = 0;
  for (int i = 0; i < prob->l; i++)
  {
    if (fabs(alpha[i]) > 0)
    {
      ++nSV;
      if (param->svm_type != SVM_PLUS)
      {
        if (prob->y[i] > 0)
        {
          if (fabs(alpha[i]) >= si.upper_bound_p)
            ++nBSV;
        }
        else
        {
          if (fabs(alpha[i]) >= si.upper_bound_n)
            ++nBSV;
        }
      }
    }
    if (param->svm_type == SVM_PLUS)
    {
      if (fabs(beta[i]) > 0)
        ++nSV_star;
    }
  }

  info("nSV = %d, nBSV = %d\n", nSV, nBSV);
  if (param->svm_type == SVM_PLUS)
    info("nSV_star = %d nBSV_star = %d \n", nSV_star, nBSV_star);

  decision_function f;
  f.alpha = alpha;
  f.rho = si.rho;
  if (param->svm_type == SVM_PLUS)
  {
    f.beta = beta;
    f.rho_star = si.rho_star;
  }

  return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
void sigmoid_train(
    int l, const double *dec_values, const double *labels,
    double &A, double &B)
{

  info("inside SIGMOID train \n");
  double prior1 = 0, prior0 = 0;
  int i;

  for (i = 0; i < l; i++)
    if (labels[i] > 0)
      prior1 += 1;
    else
      prior0 += 1;

  int max_iter = 100;      // Maximal number of iterations
  double min_step = 1e-10; // Minimal step taken in line search
  double sigma = 1e-12;    // For numerically strict PD of Hessian
  double eps = 1e-5;
  double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
  double loTarget = 1 / (prior0 + 2.0);
  double *t = Malloc(double, l);
  double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
  double newA, newB, newf, d1, d2;
  int iter;

  // Initial Point and Initial Fun Value
  A = 0.0;
  B = log((prior0 + 1.0) / (prior1 + 1.0));
  double fval = 0.0;

  for (i = 0; i < l; i++)
  {
    if (labels[i] > 0)
      t[i] = hiTarget;
    else
      t[i] = loTarget;
    fApB = dec_values[i] * A + B;
    if (fApB >= 0)
      fval += t[i] * fApB + log(1 + exp(-fApB));
    else
      fval += (t[i] - 1) * fApB + log(1 + exp(fApB));
  }
  for (iter = 0; iter < max_iter; iter++)
  {
    // Update Gradient and Hessian (use H' = H + sigma I)
    h11 = sigma; // numerically ensures strict PD
    h22 = sigma;
    h21 = 0.0;
    g1 = 0.0;
    g2 = 0.0;
    for (i = 0; i < l; i++)
    {
      fApB = dec_values[i] * A + B;
      if (fApB >= 0)
      {
        p = exp(-fApB) / (1.0 + exp(-fApB));
        q = 1.0 / (1.0 + exp(-fApB));
      }
      else
      {
        p = 1.0 / (1.0 + exp(fApB));
        q = exp(fApB) / (1.0 + exp(fApB));
      }
      d2 = p * q;
      h11 += dec_values[i] * dec_values[i] * d2;
      h22 += d2;
      h21 += dec_values[i] * d2;
      d1 = t[i] - p;
      g1 += dec_values[i] * d1;
      g2 += d1;
    }

    // Stopping Criteria
    if (fabs(g1) < eps && fabs(g2) < eps)
      break;

    // Finding Newton direction: -inv(H') * g
    det = h11 * h22 - h21 * h21;
    dA = -(h22 * g1 - h21 * g2) / det;
    dB = -(-h21 * g1 + h11 * g2) / det;
    gd = g1 * dA + g2 * dB;

    stepsize = 1; // Line Search
    while (stepsize >= min_step)
    {
      newA = A + stepsize * dA;
      newB = B + stepsize * dB;

      // New function value
      newf = 0.0;
      for (i = 0; i < l; i++)
      {
        fApB = dec_values[i] * newA + newB;
        if (fApB >= 0)
          newf += t[i] * fApB + log(1 + exp(-fApB));
        else
          newf += (t[i] - 1) * fApB + log(1 + exp(fApB));
      }
      // Check sufficient decrease
      if (newf < fval + 0.0001 * stepsize * gd)
      {
        A = newA;
        B = newB;
        fval = newf;
        break;
      }
      else
        stepsize = stepsize / 2.0;
    }

    if (stepsize < min_step)
    {
      info("Line search fails in two-class probability estimates\n");
      break;
    }
  }

  if (iter >= max_iter)
    info("Reaching maximal iterations in two-class probability estimates\n");
  free(t);
}

double sigmoid_predict(double decision_value, double A, double B)
{
  double fApB = decision_value * A + B;
  if (fApB >= 0)
    return exp(-fApB) / (1.0 + exp(-fApB));
  else
    return 1.0 / (1 + exp(fApB));
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
void multiclass_probability(int k, double **r, double *p)
{
  int t, j;
  int iter = 0, max_iter = max(100, k);
  double **Q = Malloc(double *, k);
  double *Qp = Malloc(double, k);
  double pQp, eps = 0.005 / k;

  for (t = 0; t < k; t++)
  {
    p[t] = 1.0 / k; // Valid if k = 1
    Q[t] = Malloc(double, k);
    Q[t][t] = 0;
    for (j = 0; j < t; j++)
    {
      Q[t][t] += r[j][t] * r[j][t];
      Q[t][j] = Q[j][t];
    }
    for (j = t + 1; j < k; j++)
    {
      Q[t][t] += r[j][t] * r[j][t];
      Q[t][j] = -r[j][t] * r[t][j];
    }
  }
  for (iter = 0; iter < max_iter; iter++)
  {
    // stopping condition, recalculate QP,pQP for numerical accuracy
    pQp = 0;
    for (t = 0; t < k; t++)
    {
      Qp[t] = 0;
      for (j = 0; j < k; j++)
        Qp[t] += Q[t][j] * p[j];
      pQp += p[t] * Qp[t];
    }
    double max_error = 0;
    for (t = 0; t < k; t++)
    {
      double error = fabs(Qp[t] - pQp);
      if (error > max_error)
        max_error = error;
    }
    if (max_error < eps)
      break;

    for (t = 0; t < k; t++)
    {
      double diff = (-Qp[t] + pQp) / Q[t][t];
      p[t] += diff;
      pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
      for (j = 0; j < k; j++)
      {
        Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
        p[j] /= (1 + diff);
      }
    }
  }
  if (iter >= max_iter)
    info("Exceeds max_iter in multiclass_prob\n");
  for (t = 0; t < k; t++)
    free(Q[t]);
  free(Q);
  free(Qp);
}

// Cross-validation decision values for probability estimates
void svm_binary_svc_probability(
    const svm_problem *prob, const svm_parameter *param,
    double Cp, double Cn, double &probA, double &probB)
{

  info("Inside SVM BINARY SVC PROBA\n");
  int i;
  int nr_fold = 5;
  int *perm = Malloc(int, prob->l);
  double *dec_values = Malloc(double, prob->l);

  // random shuffle
  for (i = 0; i < prob->l; i++)
    perm[i] = i;
  
  for (i = 0; i < prob->l; i++)
  {
    int j = i + rand() % (prob->l - i);
    swap(perm[i], perm[j]);
  }

  for (i = 0; i < nr_fold; i++)
  {
    int begin = i * prob->l / nr_fold;
    int end = (i + 1) * prob->l / nr_fold;
    int j, k;
    struct svm_problem subprob;

    subprob.l = prob->l - (end - begin);
    subprob.x = Malloc(struct svm_node *, subprob.l);
    subprob.y = Malloc(double, subprob.l);

    k = 0;
    for (j = 0; j < begin; j++)
    {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    for (j = end; j < prob->l; j++)
    {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    int p_count = 0, n_count = 0;
    for (j = 0; j < k; j++)
      if (subprob.y[j] > 0)
        p_count++;
      else
        n_count++;

    if (p_count == 0 && n_count == 0)
      for (j = begin; j < end; j++)
        dec_values[perm[j]] = 0;
    else if (p_count > 0 && n_count == 0)
      for (j = begin; j < end; j++)
        dec_values[perm[j]] = 1;
    else if (p_count == 0 && n_count > 0)
      for (j = begin; j < end; j++)
        dec_values[perm[j]] = -1;
    else
    {
      
      svm_parameter subparam = *param;
      subparam.probability = 0;
      subparam.C = 1.0;
      subparam.nr_weight = 2;
      subparam.weight_label = Malloc(int, 2);
      subparam.weight = Malloc(double, 2);
      subparam.weight_label[0] = +1;
      subparam.weight_label[1] = -1;
      subparam.weight[0] = Cp;
      subparam.weight[1] = Cn;
      
      struct svm_model *submodel = svm_train(&subprob, &subparam);
      
      for (j = begin; j < end; j++)
      {
        svm_predict_values(submodel, prob->x[perm[j]], &(dec_values[perm[j]]));
        // ensure +1 -1 order; reason not using CV subroutine
        dec_values[perm[j]] *= submodel->label[0];
      }
      svm_destroy_model(submodel);
      svm_destroy_param(&subparam);
    }

    free(subprob.x);
    free(subprob.y);
  }

  sigmoid_train(prob->l, dec_values, prob->y, probA, probB);
  free(dec_values);
  free(perm);
}

// Return parameter of a Laplace distribution
double svm_svr_probability(
    const svm_problem *prob, const svm_parameter *param)
{
  int i;
  int nr_fold = 5;
  double *ymv = Malloc(double, prob->l);
  double mae = 0;

  svm_parameter newparam = *param;
  newparam.probability = 0;
  svm_cross_validation(prob, &newparam, nr_fold, ymv);
  for (i = 0; i < prob->l; i++)
  {
    ymv[i] = prob->y[i] - ymv[i];
    mae += fabs(ymv[i]);
  }
  mae /= prob->l;
  double std = sqrt(2 * mae * mae);
  int count = 0;
  mae = 0;
  for (i = 0; i < prob->l; i++)
    if (fabs(ymv[i]) > 5 * std)
      count = count + 1;
    else
      mae += fabs(ymv[i]);
  mae /= (prob->l - count);
  info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n", mae);
  free(ymv);
  return mae;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
  int l = prob->l;
  int max_nr_class = 16;
  int nr_class = 0;
  int *label = Malloc(int, max_nr_class);
  int *count = Malloc(int, max_nr_class);
  int *data_label = Malloc(int, l);
  int i;

  for (i = 0; i < l; i++)
  {
    int this_label = (int)prob->y[i];
    int j;
    for (j = 0; j < nr_class; j++)
    {
      if (this_label == label[j])
      {
        ++count[j];
        break;
      }
    }
    data_label[i] = j;
    if (j == nr_class)
    {
      if (nr_class == max_nr_class)
      {
        max_nr_class *= 2;
        label = (int *)realloc(label, max_nr_class * sizeof(int));
        count = (int *)realloc(count, max_nr_class * sizeof(int));
      }
      label[nr_class] = this_label;
      count[nr_class] = 1;
      ++nr_class;
    }
  }

  int *start = Malloc(int, nr_class);
  start[0] = 0;
  for (i = 1; i < nr_class; i++)
    start[i] = start[i - 1] + count[i - 1];
  for (i = 0; i < l; i++)
  {
    perm[start[data_label[i]]] = i;
    ++start[data_label[i]];
  }
  start[0] = 0;
  for (i = 1; i < nr_class; i++)
    start[i] = start[i - 1] + count[i - 1];

  *nr_class_ret = nr_class;
  *label_ret = label;
  *start_ret = start;
  *count_ret = count;
  free(data_label);
}
//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
  
  // info("Doing svm_train \n");

  svm_model *model = Malloc(svm_model, 1);
  model->param = *param;
  model->free_sv = 0; // XXX

  if (param->svm_type == ONE_CLASS ||
      param->svm_type == EPSILON_SVR ||
      param->svm_type == NU_SVR)
  {
    // regression or one-class-svm
    model->nr_class = 2;
    model->label = NULL;
    model->nSV = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->sv_coef = Malloc(double *, 1);

    if (param->probability &&
        (param->svm_type == EPSILON_SVR ||
         param->svm_type == NU_SVR))
    {
      model->probA = Malloc(double, 1);
      model->probA[0] = svm_svr_probability(prob, param);
    }

    decision_function f = svm_train_one(prob, param, 0, 0);
    model->rho = Malloc(double, 1);
    model->rho[0] = f.rho;

    int nSV = 0;
    int i;
    for (i = 0; i < prob->l; i++)
      if (fabs(f.alpha[i]) > 0)
        ++nSV;
    model->l = nSV;
    model->SV = Malloc(svm_node *, nSV);
    model->sv_coef[0] = Malloc(double, nSV);
    int j = 0;
    for (i = 0; i < prob->l; i++)
      if (fabs(f.alpha[i]) > 0)
      {
        model->SV[j] = prob->x[i];
        model->sv_coef[0][j] = f.alpha[i];
        ++j;
      }

    free(f.alpha);
  }
  else
  {
    
    // classification
    int l = prob->l;
    int nr_class;
    int *label = NULL;
    int *start = NULL;
    int *count = NULL;
    int *perm = Malloc(int, l);

    /*XXX*/

    // group training data of the same class
    svm_group_classes(prob, &nr_class, &label, &start, &count, perm);
    

    svm_node **x = Malloc(svm_node *, l);
    svm_node **x_star = NULL;
    // x_star = Malloc(svm_node *, l);
    int i;

    for (i = 0; i < l; i++)
      x[i] = prob->x[perm[i]];

    if (param->svm_type == SVM_PLUS)
    {

      x_star = Malloc(svm_node *, l);
      for (i = 0; i < l; i++)
      {     
        x_star[i] = prob->x_star[perm[i]];
        //info("x_star = %f\n", x_star[i]);
      }
    }
    // calculate weighted C
    double *weighted_C = Malloc(double, nr_class);
    for (i = 0; i < nr_class; i++)
      weighted_C[i] = param->C;


    for (i = 0; i < param->nr_weight; i++)
    {
      int j;
      for (j = 0; j < nr_class; j++)
        if (param->weight_label[i] == label[j])
          break;
      if (j == nr_class)
        info("warning: class label %d specified in weight is not found\n", param->weight_label[i]);
        
      else
        weighted_C[j] *= param->weight[i];
    }

    // train k*(k-1)/2 models
    bool *nonzero = Malloc(bool, l);
    bool *nonzero_star = Malloc(bool, l);

    for (i = 0; i < l; i++)
    {
      nonzero[i] = false;
      nonzero_star[i] = false;
    }

    decision_function *f = Malloc(decision_function, nr_class * (nr_class - 1) / 2);

    double *probA = NULL, *probB = NULL;

    if (param->probability)
    {
      probA = Malloc(double, nr_class *(nr_class - 1) / 2);
      probB = Malloc(double, nr_class *(nr_class - 1) / 2);
    }

    int p = 0, p_star = 0;

    for (i = 0; i < nr_class; i++)
      for (int j = i + 1; j < nr_class; j++)
      {
        svm_problem sub_prob;
        int si = start[i], sj = start[j];
        int ci = count[i], cj = count[j];
        sub_prob.l = ci + cj;
        sub_prob.x = Malloc(svm_node *, sub_prob.l);

        if (param->svm_type == SVM_PLUS)
          sub_prob.x_star = Malloc(svm_node *, sub_prob.l);
        sub_prob.y = Malloc(double, sub_prob.l);
        int k;

        for (k = 0; k < ci; k++)
        {
          sub_prob.x[k] = x[si + k];
          sub_prob.y[k] = +1;
          if (param->svm_type == SVM_PLUS)
            sub_prob.x_star[k] = x_star[si + k];
        }
        for (k = 0; k < cj; k++)
        {
          sub_prob.x[ci + k] = x[sj + k];
          sub_prob.y[ci + k] = -1;
          if (param->svm_type == SVM_PLUS)
            sub_prob.x_star[ci + k] = x_star[sj + k];
        }

        if (param->probability)
          svm_binary_svc_probability(&sub_prob, param, weighted_C[i], weighted_C[j], probA[p], probB[p]);

        f[p] = svm_train_one(&sub_prob, param, weighted_C[i], weighted_C[j]);

        /*
        if (param->svm_type == SVM_PLUS)
          for (int i = 0; i < prob->l; i++)
          {
            info("%f %f\n", f[p].alpha[i], f[p].beta[i]);
            
          }
          */
        //fflush(stdout);

        for (k = 0; k < ci; k++)
        {
          if (!nonzero[si + k] && fabs(f[p].alpha[k]) > 0)
            nonzero[si + k] = true;
          if (param->svm_type == SVM_PLUS)
            if (!nonzero_star[si + k] && f[p].beta[k] > 0)
              nonzero_star[si + k] = true;
        }

        for (k = 0; k < cj; k++)
        {
          if (!nonzero[sj + k] && fabs(f[p].alpha[ci + k]) > 0)
            nonzero[sj + k] = true;
          if (param->svm_type == SVM_PLUS)
            if (!nonzero_star[sj + k] && f[p].beta[ci + k] > 0)
              nonzero_star[sj + k] = true;
        }

        free(sub_prob.x);

        if (param->svm_type == SVM_PLUS)
          free(sub_prob.x_star);
        free(sub_prob.y);
        ++p;
      }

    // build output
    model->nr_class = nr_class;

    model->label = Malloc(int, nr_class);
    for (i = 0; i < nr_class; i++)
      model->label[i] = label[i];

    model->rho = Malloc(double, nr_class *(nr_class - 1) / 2);
    for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
      model->rho[i] = f[i].rho;

    if (param->svm_type == SVM_PLUS)
    {
      model->rho_star = Malloc(double, nr_class *(nr_class - 1) / 2);
      for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
        model->rho_star[i] = f[i].rho_star;
    }

    if (param->probability)
    {
      model->probA = Malloc(double, nr_class *(nr_class - 1) / 2);
      model->probB = Malloc(double, nr_class *(nr_class - 1) / 2);
      for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
      {
        model->probA[i] = probA[i];
        model->probB[i] = probB[i];
      }
    }
    else
    {
      model->probA = NULL;
      model->probB = NULL;
    }

    int total_sv = 0;
    int total_sv_star = 0;
    int *nz_count = Malloc(int, nr_class);
    int *nz_count_star = Malloc(int, nr_class);
    model->nSV = Malloc(int, nr_class);
    model->nSV_star = Malloc(int, nr_class);
    for (i = 0; i < nr_class; i++)
    {
      int nSV = 0;
      int nSV_star = 0;
      for (int j = 0; j < count[i]; j++)
      {
        if (nonzero[start[i] + j])
        {
          ++nSV;
          ++total_sv;
        }
        if (nonzero_star[start[i] + j])
        {
          ++nSV_star;
          ++total_sv_star;
        }
      }
      model->nSV[i] = nSV;
      nz_count[i] = nSV;
      model->nSV_star[i] = nSV_star;
      nz_count_star[i] = nSV_star;
    }

    info("Total nSV = %d\n", total_sv);
    info("Total nSV_star = %d\n", total_sv_star);

    model->l = total_sv;
    model->SV = Malloc(svm_node *, total_sv);
    p = 0;
    for (i = 0; i < l; i++)
      if (nonzero[i])
        model->SV[p++] = x[i];

    if (model->param.svm_type == SVM_PLUS)
    {
      model->SV_star = Malloc(svm_node *, total_sv_star);
      p_star = 0;
      for (i = 0; i < l; i++)
        if (nonzero_star[i])
          model->SV_star[p_star++] = x_star[i];
    }

    int *nz_start = Malloc(int, nr_class);
    nz_start[0] = 0;
    for (i = 1; i < nr_class; i++)
      nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

    model->sv_coef = Malloc(double *, nr_class - 1);
    for (i = 0; i < nr_class - 1; i++)
      model->sv_coef[i] = Malloc(double, total_sv);

    int *nz_start_star = NULL;
    if (model->param.svm_type == SVM_PLUS)
    {
      nz_start_star = Malloc(int, nr_class);
      nz_start_star[0] = 0;
      for (i = 1; i < nr_class; i++)
        nz_start_star[i] = nz_start_star[i - 1] + nz_count_star[i - 1];

      model->sv_coef_star = Malloc(double *, nr_class - 1);
      for (i = 0; i < nr_class - 1; i++)
        model->sv_coef_star[i] = Malloc(double, total_sv_star);
    }

    p = 0;
    for (i = 0; i < nr_class; i++)
      for (int j = i + 1; j < nr_class; j++)
      {
        // classifier (i,j): coefficients with
        // i are in sv_coef[j-1][nz_start[i]...],
        // j are in sv_coef[i][nz_start[j]...]

        int si = start[i];
        int sj = start[j];
        int ci = count[i];
        int cj = count[j];

        int q = nz_start[i];
        int q_star = 0;
        if (model->param.svm_type == SVM_PLUS)
          q_star = nz_start_star[i];
        int k;
        for (k = 0; k < ci; k++)
        {
          if (nonzero[si + k])
            model->sv_coef[j - 1][q++] = f[p].alpha[k];
          if (model->param.svm_type == SVM_PLUS)
            if (nonzero_star[si + k])
              model->sv_coef_star[j - 1][q_star++] = f[p].beta[k];
        }
        q = nz_start[j];
        if (model->param.svm_type == SVM_PLUS)
          q_star = nz_start_star[j];
        for (k = 0; k < cj; k++)
        {
          if (nonzero[sj + k])
            model->sv_coef[i][q++] = f[p].alpha[ci + k];
          if (model->param.svm_type == SVM_PLUS)
            if (nonzero_star[sj + k])
              model->sv_coef_star[i][q_star++] = f[p].beta[ci + k];
        }
        ++p;
      }

      /*if(model->param.svm_type == SVM_PLUS)
            for(int i=0; i<model->l; i++)
              info("%f %f\n", model->sv_coef[0][i], model->sv_coef_star[0][i]);
              //fprintf(stdout,"%f %f\n", model->sv_coef[0][i], model->sv_coef_star[0][i]);
          //fflush(stdout);
          */
    
    // commented this section because it causes core dumped issue in cygwin
    // Okba BEKHELIFI
    free(label);
    free(probA);
    free(probB);
    free(count);
    free(perm);
    free(start);
    free(x);
    free(weighted_C);
    free(nonzero);
    for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
      free(f[i].alpha);
    
    free(nz_count);
    free(nz_start);
    // free(f);
    
    if (model->param.svm_type == SVM_PLUS)
    {
      free(x_star);
      for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
        free(f[i].beta);
      free(f);
      free(nz_count_star);
      free(nz_start_star);
    }
    
  }
  
  // info("Returning Model\n");
  return model;
}

/* Broken */
// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{

  info("// Stratified cross validation\n");

  int i;

  int *fold_start = Malloc(int, nr_fold + 1);
  int l = prob->l;
  int *perm = Malloc(int, l);
  int nr_class;
  if (nr_fold > l)
  {
    nr_fold = l;
    info("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
  }

  // stratified cv may not give leave-one-out rate
  // Each class to l folds -> some folds may have zero elements
  info(" About to test params \n");
  info(" svm type: %d\n", param->svm_type);
  info(" nr fold %d\n", nr_fold);

  if ((param->svm_type == C_SVC ||
       param->svm_type == NU_SVC) &&
      nr_fold < l)
  {
    info("Doing CV for C_SVC or NU_SVC");
    int *start = NULL;
    int *label = NULL;
    int *count = NULL;
    svm_group_classes(prob, &nr_class, &label, &start, &count, perm);

    // random shuffle and then data grouped by fold using the array perm
    int *fold_count = Malloc(int, nr_fold);
    int c;
    int *index = Malloc(int, l);

    for (i = 0; i < l; i++)
      index[i] = perm[i];

    for (c = 0; c < nr_class; c++)
      for (i = 0; i < count[c]; i++)
      {
        int j = i + rand() % (count[c] - i);
        swap(index[start[c] + j], index[start[c] + i]);
      }

    for (i = 0; i < nr_fold; i++)
    {
      fold_count[i] = 0;
      for (c = 0; c < nr_class; c++)
        fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold;
    }
    fold_start[0] = 0;

    for (i = 1; i <= nr_fold; i++)
      fold_start[i] = fold_start[i - 1] + fold_count[i - 1];

    for (c = 0; c < nr_class; c++)
      for (i = 0; i < nr_fold; i++)
      {
        int begin = start[c] + i * count[c] / nr_fold;
        int end = start[c] + (i + 1) * count[c] / nr_fold;
        for (int j = begin; j < end; j++)
        {
          perm[fold_start[i]] = index[j];
          fold_start[i]++;
        }
      }

    fold_start[0] = 0;

    for (i = 1; i <= nr_fold; i++)
      fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
    free(start);
    free(label);
    free(count);
    free(index);
    free(fold_count);
  }

  else
  {
    info("Doing CV for types different than C_SVC or NU_SVC\n");
    for (i = 0; i < l; i++)
      perm[i] = i;
    for (i = 0; i < l; i++)
    {
      int j = i + rand() % (l - i);
      swap(perm[i], perm[j]);
    }
    for (i = 0; i <= nr_fold; i++)
      fold_start[i] = i * l / nr_fold;
  }

  for (i = 0; i < nr_fold; i++)
  {
    int begin = fold_start[i];
    int end = fold_start[i + 1];
    int j, k;
    struct svm_problem subprob;

    subprob.l = l - (end - begin);
    subprob.x = Malloc(struct svm_node *, subprob.l);
    subprob.y = Malloc(double, subprob.l);

    k = 0;
    for (j = 0; j < begin; j++)
    {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    for (j = end; j < l; j++)
    {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    info("Training a sub model \n");
    struct svm_model *submodel = svm_train(&subprob, param);
    info("Sub model trained successfully \n");

    if (param->probability &&
        (param->svm_type == C_SVC || param->svm_type == NU_SVC))
    {
      double *prob_estimates = Malloc(double, svm_get_nr_class(submodel));
      for (j = begin; j < end; j++)
        target[perm[j]] = svm_predict_probability(submodel, prob->x[perm[j]], prob_estimates);
      free(prob_estimates);
    }
    else
      for (j = begin; j < end; j++)
        target[perm[j]] = svm_predict(submodel, prob->x[perm[j]]);
    svm_destroy_model(submodel);
    free(subprob.x);
    free(subprob.y);
  }
  free(fold_start);
  free(perm);
}

int svm_get_svm_type(const svm_model *model)
{
  return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
  return model->nr_class;
}

void svm_get_labels(const svm_model *model, int *label)
{
  if (model->label != NULL)
    for (int i = 0; i < model->nr_class; i++)
      label[i] = model->label[i];
}

double svm_get_svr_probability(const svm_model *model)
{
  if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
      model->probA != NULL)
    return model->probA[0];
  else
  {
    fprintf(stderr, "Model doesn't contain information for SVR probability inference\n");
    return 0;
  }
}

void svm_predict_values(const svm_model *model, const svm_node *x, double *dec_values)
{
  info("inside SVM PREDICT VALUES \n");

  if (model->param.svm_type == ONE_CLASS ||
      model->param.svm_type == EPSILON_SVR ||
      model->param.svm_type == NU_SVR)
  {
    double *sv_coef = model->sv_coef[0];
    double sum = 0;
    for (int i = 0; i < model->l; i++)
      sum += sv_coef[i] * Kernel::k_function(x, model->SV[i], model->param);
    sum -= model->rho[0];
    *dec_values = sum;
  }
  else
  {
    int i;
    int nr_class = model->nr_class;
    int l = model->l;

    double *kvalue = Malloc(double, l);
    for (i = 0; i < l; i++)
      kvalue[i] = Kernel::k_function(x, model->SV[i], model->param);

    int *start = Malloc(int, nr_class);
    start[0] = 0;
    for (i = 1; i < nr_class; i++)
      start[i] = start[i - 1] + model->nSV[i - 1];

    int p = 0;
    for (i = 0; i < nr_class; i++)
      for (int j = i + 1; j < nr_class; j++)
      {
        double sum = 0;
        int si = start[i];
        int sj = start[j];
        int ci = model->nSV[i];
        int cj = model->nSV[j];

        int k;
        double *coef1 = model->sv_coef[j - 1];
        double *coef2 = model->sv_coef[i];
        for (k = 0; k < ci; k++)
          sum += coef1[si + k] * kvalue[si + k];
        for (k = 0; k < cj; k++)
          sum += coef2[sj + k] * kvalue[sj + k];
        sum -= model->rho[p];
        dec_values[p] = sum;
        p++;
      }

    free(kvalue);
    free(start);
  }
}

double svm_predict(const svm_model *model, const svm_node *x)
{
  info("Inside svm_predict \n");
  if (model->param.svm_type == ONE_CLASS ||
      model->param.svm_type == EPSILON_SVR ||
      model->param.svm_type == NU_SVR)
  {
    double res;
    svm_predict_values(model, x, &res);

    if (model->param.svm_type == ONE_CLASS)
      return (res > 0) ? 1 : -1;
    else
      return res;
  }
  else
  {
    int i;
    int nr_class = model->nr_class;
    double *dec_values = Malloc(double, nr_class *(nr_class - 1) / 2);
    svm_predict_values(model, x, dec_values);
    double soft_classification = dec_values[0] * model->label[0];

    //printf("cl in svm_predict %f\n",soft_classification);

    int *vote = Malloc(int, nr_class);
    for (i = 0; i < nr_class; i++)
      vote[i] = 0;
    int pos = 0;
    for (i = 0; i < nr_class; i++)
      for (int j = i + 1; j < nr_class; j++)
      {
        if (dec_values[pos++] > 0)
          ++vote[i];
        else
          ++vote[j];
      }

    int vote_max_idx = 0;
    for (i = 1; i < nr_class; i++)
      if (vote[i] > vote[vote_max_idx])
        vote_max_idx = i;

    /*
    info("vote max: %d\n", vote_max_idx);
    mexPrintf("soft_classification %f\n", soft_classification);
    mexPrintf("vote max: %d\n", vote_max_idx);
    mexPrintf("label vote max: %d\n", model->label[vote_max_idx]);
    */
    free(vote);
    free(dec_values);
    // return soft_classification;
    return model->label[vote_max_idx];
  }
}

double svm_predict_probability(
    const svm_model *model, const svm_node *x, double *prob_estimates)
{
  if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
      model->probA != NULL && model->probB != NULL)
  {
    int i;
    int nr_class = model->nr_class;
    double *dec_values = Malloc(double, nr_class *(nr_class - 1) / 2);
    svm_predict_values(model, x, dec_values);

    double min_prob = 1e-7;
    double **pairwise_prob = Malloc(double *, nr_class);
    for (i = 0; i < nr_class; i++)
      pairwise_prob[i] = Malloc(double, nr_class);
    int k = 0;
    for (i = 0; i < nr_class; i++)
      for (int j = i + 1; j < nr_class; j++)
      {
        pairwise_prob[i][j] = min(max(sigmoid_predict(dec_values[k], model->probA[k], model->probB[k]), min_prob), 1 - min_prob);
        pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
        k++;
      }
    multiclass_probability(nr_class, pairwise_prob, prob_estimates);

    int prob_max_idx = 0;
    for (i = 1; i < nr_class; i++)
      if (prob_estimates[i] > prob_estimates[prob_max_idx])
        prob_max_idx = i;
    for (i = 0; i < nr_class; i++)
      free(pairwise_prob[i]);
    free(dec_values);
    free(pairwise_prob);
    return model->label[prob_max_idx];
  }
  else
    return svm_predict(model, x);
}

const char *svm_type_table[] =
    {
        "c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", "svmp_plus", NULL};

const char *kernel_type_table[] =
    {
        "linear", "polynomial", "rbf", "sigmoid", "precomputed", NULL};













void svm_destroy_param(svm_parameter *param)
{
  free(param->weight_label);
  free(param->weight);
}
void svm_destroy_model(svm_model *model)
{
  
  // info("inside svm destroy model\n");
  int i;
  // info("free_sv %d\n", model->free_sv);
  //info("l %d\n", model->l);
/*
  if(model->free_sv && model->l > 0) {
    //   free((void *)(model->SV[0]));
   //    free((model->SV[0]));
    /*
      if(model->param.svm_type == SVM_PLUS)
      free((void *)(model->SV_star[0]));
    
  }
  for(i=0;i<model->nr_class-1;i++) 
    free(model->sv_coef[i]);
  /*
    if(model->param.svm_type == SVM_PLUS) {
    for(i=0;i<model->nr_class-1;i++) 
    free(model->sv_coef[i]);
    free(model->rho_star);
    free(model->SV_star);
    free(model->sv_coef_star);
    free(model->nSV_star);
    }
  
  free(model->SV);
  // free(model->sv_coef);
  free(model->rho);
  free(model->label);
  free(model->probA);
  free(model->probB);
  free(model->nSV);
free(model);
*/
}
const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
  // svm_type

  int svm_type = param->svm_type;
  if (svm_type != C_SVC &&
      svm_type != NU_SVC &&
      svm_type != ONE_CLASS &&
      svm_type != EPSILON_SVR &&
      svm_type != NU_SVR &&
      svm_type != SVM_PLUS)
    return "unknown svm type";

  // kernel_type, degree

  int kernel_type = param->kernel_type;
  if (kernel_type != LINEAR &&
      kernel_type != POLY &&
      kernel_type != RBF &&
      kernel_type != SIGMOID &&
      kernel_type != PRECOMPUTED)
    return "unknown kernel type";

  int kernel_type_star = param->kernel_type_star;
  if (kernel_type_star != LINEAR &&
      kernel_type_star != POLY &&
      kernel_type_star != RBF &&
      kernel_type_star != SIGMOID &&
      kernel_type_star != PRECOMPUTED)
    return "unknown kernel type for the correcting space";

  if (param->degree < 0)
    return "degree of polynomial kernel < 0";

  if (param->degree_star < 0)
    return "degree of polynomial kernel for the correcting space < 0";

  // cache_size,eps,C,tau,nu,p,shrinking

  if (param->cache_size <= 0)
    return "cache_size <= 0";

  if (param->eps <= 0)
    return "eps <= 0";

  if (svm_type == C_SVC ||
      svm_type == SVM_PLUS ||
      svm_type == EPSILON_SVR ||
      svm_type == NU_SVR)
    if (param->C <= 0)
      return "C <= 0";

  if (svm_type == SVM_PLUS)
    if (param->tau <= 0)
      return "tau <= 0";

  if (svm_type == NU_SVC ||
      svm_type == ONE_CLASS ||
      svm_type == NU_SVR)
    if (param->nu <= 0 || param->nu > 1)
      return "nu <= 0 or nu > 1";

  if (svm_type == EPSILON_SVR)
    if (param->p < 0)
      return "p < 0";

  if (param->shrinking != 0 &&
      param->shrinking != 1)
    return "shrinking != 0 and shrinking != 1";

  if (param->probability != 0 &&
      param->probability != 1)
    return "probability != 0 and probability != 1";

  if (param->probability == 1 &&
      svm_type == ONE_CLASS)
    return "one-class SVM probability output not supported yet";

  // check whether nu-svc is feasible

  if (svm_type == NU_SVC)
  {
    int l = prob->l;
    int max_nr_class = 16;
    int nr_class = 0;
    int *label = Malloc(int, max_nr_class);
    int *count = Malloc(int, max_nr_class);

    int i;
    for (i = 0; i < l; i++)
    {
      int this_label = (int)prob->y[i];
      int j;
      for (j = 0; j < nr_class; j++)
        if (this_label == label[j])
        {
          ++count[j];
          break;
        }
      if (j == nr_class)
      {
        if (nr_class == max_nr_class)
        {
          max_nr_class *= 2;
          label = (int *)realloc(label, max_nr_class * sizeof(int));
          count = (int *)realloc(count, max_nr_class * sizeof(int));
        }
        label[nr_class] = this_label;
        count[nr_class] = 1;
        ++nr_class;
      }
    }

    for (i = 0; i < nr_class; i++)
    {
      int n1 = count[i];
      for (int j = i + 1; j < nr_class; j++)
      {
        int n2 = count[j];
        if (param->nu * (n1 + n2) / 2 > min(n1, n2))
        {
          free(label);
          free(count);
          return "specified nu is infeasible";
        }
      }
    }
    free(label);
    free(count);
  }

  return NULL;
}

int check_compatibility(struct svm_problem prob, struct svm_problem prob_star)
{
  int i;

  if (prob.l != prob_star.l)
  {
    info("Different number of examples in X and X* space\n");
    // exit(1);
    return 1;
  }

  for (i = 0; i < prob.l; i++)
    if (prob.y[i] != prob_star.y[i])
    {
      info("prob.y[i] = %d prob_star.y[i]= %d\n", prob.y[i], prob_star.y[i]);
      info("Different labels in example %d\n", i);
      // exit(1);
      return 1;
    }
}

void check_kernel_input(struct svm_problem prob, int max_index)
{
  int i;
  for (i = 0; i < prob.l; i++)
  {
    if (prob.x[i][0].index != 0)
    {
      fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
      exit(1);
    }
    if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
    {
      fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
      exit(1);
    }
  }
}

int svm_check_probability_model(const svm_model *model)
{
  return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
          model->probA != NULL && model->probB != NULL) ||
         ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
          model->probA != NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
  if (print_func == NULL)
    svm_print_string = &print_string_stdout;
  else
    svm_print_string = print_func;
}



/* VERBOSE */
// TODO : replace or remove
/*
int svm_save_model(const char *model_file_name, const svm_model *model)
{
  FILE *fp = fopen(model_file_name, "w");
  fprintf(stdout, "model_file_name=%s", model_file_name);
  fflush(stdout);
  if (fp == NULL)
    return -1;

  const svm_parameter &param = model->param;

  fprintf(fp, "svm_type %s\n", svm_type_table[param.svm_type]);
  fprintf(fp, "kernel_type %s\n", kernel_type_table[param.kernel_type]);

  if (param.kernel_type == POLY)
    fprintf(fp, "degree %d\n", param.degree);

  if (param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
    fprintf(fp, "gamma %g\n", param.gamma);

  if (param.kernel_type == POLY || param.kernel_type == SIGMOID)
    fprintf(fp, "coef0 %g\n", param.coef0);

  int nr_class = model->nr_class;
  int l = model->l;
  fprintf(fp, "nr_class %d\n", nr_class);
  fprintf(fp, "total_sv %d\n", l);
  fprintf(stdout, "total_sv=%d", l);
  fflush(stdout);

  {
    fprintf(fp, "rho");
    for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
      fprintf(fp, " %g", model->rho[i]);
    fprintf(fp, "\n");
  }

  if (model->label)
  {
    fprintf(fp, "label");
    for (int i = 0; i < nr_class; i++)
      fprintf(fp, " %d", model->label[i]);
    fprintf(fp, "\n");
  }

  if (model->probA) // regression has probA only
  {
    fprintf(fp, "probA");
    for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
      fprintf(fp, " %g", model->probA[i]);
    fprintf(fp, "\n");
  }
  if (model->probB)
  {
    fprintf(fp, "probB");
    for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
      fprintf(fp, " %g", model->probB[i]);
    fprintf(fp, "\n");
  }

  if (model->nSV)
  {
    fprintf(fp, "nr_sv");
    for (int i = 0; i < nr_class; i++)
      fprintf(fp, " %d", model->nSV[i]);
    fprintf(fp, "\n");
  }

  fprintf(fp, "SV\n");
  const double *const *sv_coef = model->sv_coef;
  const svm_node *const *SV = model->SV;

  for (int i = 0; i < l; i++)
  {
    for (int j = 0; j < nr_class - 1; j++)
    {
      fprintf(fp, "%.16g ", sv_coef[j][i]);
      fflush(stdout);
    }
    const svm_node *p = SV[i];

    if (param.kernel_type == PRECOMPUTED)
      fprintf(fp, "0:%d ", (int)(p->value));
    else
      while (p->index != -1)
      {
        fprintf(fp, "%d:%.8g ", p->index, p->value);
        p++;
      }
    fprintf(fp, "\n");
  }
  exit(1);

  if (ferror(fp) != 0 || fclose(fp) != 0)
    return -1;
  else
    return 0;
}
*/

/* static char *line = NULL;
static int max_line_len;
/* VERBOSE */
/* static char *readline(FILE *input)
{
  int len;

  if (fgets(line, max_line_len, input) == NULL)
    return NULL;

  while (strrchr(line, '\n') == NULL)
  {
    max_line_len *= 2;
    line = (char *)realloc(line, max_line_len);
    len = (int)strlen(line);
    if (fgets(line + len, max_line_len - len, input) == NULL)
      break;
  }
  return line;
}
*/
/* VERBOSE */
//TODO: replace by read_model_header
/* svm_model *svm_load_model(const char *model_file_name)
{
  FILE *fp = fopen(model_file_name, "rb");
  if (fp == NULL)
    return NULL;

  // read parameters
  svm_model *model = Malloc(svm_model, 1);
  svm_parameter &param = model->param;
  model->rho = NULL;
  model->probA = NULL;
  model->probB = NULL;
  model->label = NULL;
  model->nSV = NULL;

  char cmd[81];
  while (1)
  {
    fscanf(fp, "%80s", cmd);

    if (strcmp(cmd, "svm_type") == 0)
    {
      fscanf(fp, "%80s", cmd);
      int i;
      for (i = 0; svm_type_table[i]; i++)
      {
        if (strcmp(svm_type_table[i], cmd) == 0)
        {
          param.svm_type = i;
          break;
        }
      }
      if (svm_type_table[i] == NULL)
      {
        fprintf(stderr, "unknown svm type.\n");
        free(model->rho);
        free(model->label);
        free(model->nSV);
        free(model);
        return NULL;
      }
    }
    else if (strcmp(cmd, "kernel_type") == 0)
    {
      fscanf(fp, "%80s", cmd);
      int i;
      for (i = 0; kernel_type_table[i]; i++)
      {
        if (strcmp(kernel_type_table[i], cmd) == 0)
        {
          param.kernel_type = i;
          break;
        }
      }
      if (kernel_type_table[i] == NULL)
      {
        fprintf(stderr, "unknown kernel function.\n");
        free(model->rho);
        free(model->label);
        free(model->nSV);
        free(model);
        return NULL;
      }
    }
    else if (strcmp(cmd, "degree") == 0)
      fscanf(fp, "%d", &param.degree);
    else if (strcmp(cmd, "gamma") == 0)
      fscanf(fp, "%lf", &param.gamma);
    else if (strcmp(cmd, "coef0") == 0)
      fscanf(fp, "%lf", &param.coef0);
    else if (strcmp(cmd, "nr_class") == 0)
      fscanf(fp, "%d", &model->nr_class);
    else if (strcmp(cmd, "total_sv") == 0)
      fscanf(fp, "%d", &model->l);
    else if (strcmp(cmd, "rho") == 0)
    {
      int n = model->nr_class * (model->nr_class - 1) / 2;
      model->rho = Malloc(double, n);
      for (int i = 0; i < n; i++)
        fscanf(fp, "%lf", &model->rho[i]);
    }
    else if (strcmp(cmd, "label") == 0)
    {
      int n = model->nr_class;
      model->label = Malloc(int, n);
      for (int i = 0; i < n; i++)
        fscanf(fp, "%d", &model->label[i]);
    }
    else if (strcmp(cmd, "probA") == 0)
    {
      int n = model->nr_class * (model->nr_class - 1) / 2;
      model->probA = Malloc(double, n);
      for (int i = 0; i < n; i++)
        fscanf(fp, "%lf", &model->probA[i]);
    }
    else if (strcmp(cmd, "probB") == 0)
    {
      int n = model->nr_class * (model->nr_class - 1) / 2;
      model->probB = Malloc(double, n);
      for (int i = 0; i < n; i++)
        fscanf(fp, "%lf", &model->probB[i]);
    }
    else if (strcmp(cmd, "nr_sv") == 0)
    {
      int n = model->nr_class;
      model->nSV = Malloc(int, n);
      for (int i = 0; i < n; i++)
        fscanf(fp, "%d", &model->nSV[i]);
    }
    else if (strcmp(cmd, "SV") == 0)
    {
      while (1)
      {
        int c = getc(fp);
        if (c == EOF || c == '\n')
          break;
      }
      break;
    }
    else
    {
      fprintf(stderr, "unknown text in model file: [%s]\n", cmd);
      free(model->rho);
      free(model->label);
      free(model->nSV);
      free(model);
      return NULL;
    }
  }

  // read sv_coef and SV

  int elements = 0;
  long pos = ftell(fp);

  max_line_len = 1024;
  line = Malloc(char, max_line_len);
  char *p, *endptr, *idx, *val;

  while (readline(fp) != NULL)
  {
    p = strtok(line, ":");
    while (1)
    {
      p = strtok(NULL, ":");
      if (p == NULL)
        break;
      ++elements;
    }
  }
  elements += model->l;

  fseek(fp, pos, SEEK_SET);

  int m = model->nr_class - 1;
  int l = model->l;
  model->sv_coef = Malloc(double *, m);
  int i;
  for (i = 0; i < m; i++)
    model->sv_coef[i] = Malloc(double, l);
  model->SV = Malloc(svm_node *, l);
  svm_node *x_space = NULL;
  if (l > 0)
    x_space = Malloc(svm_node, elements);

  int j = 0;
  for (i = 0; i < l; i++)
  {
    readline(fp);
    model->SV[i] = &x_space[j];

    p = strtok(line, " \t");
    model->sv_coef[0][i] = strtod(p, &endptr);
    for (int k = 1; k < m; k++)
    {
      p = strtok(NULL, " \t");
      model->sv_coef[k][i] = strtod(p, &endptr);
    }

    while (1)
    {
      idx = strtok(NULL, ":");
      val = strtok(NULL, " \t");

      if (val == NULL)
        break;
      x_space[j].index = (int)strtol(idx, &endptr, 10);
      x_space[j].value = strtod(val, &endptr);

      ++j;
    }
    x_space[j++].index = -1;
  }
  free(line);

  if (ferror(fp) != 0 || fclose(fp) != 0)
    return NULL;

  model->free_sv = 1; // XXX
  return model;
}

void svm_destroy_model(svm_model *model)
{
  
  // info("inside svm destroy model\n");
  int i;
  // info("free_sv %d\n", model->free_sv);
  //info("l %d\n", model->l);
/*
  if(model->free_sv && model->l > 0) {
    //   free((void *)(model->SV[0]));
   //    free((model->SV[0]));
    /*
      if(model->param.svm_type == SVM_PLUS)
      free((void *)(model->SV_star[0]));
    
  }
  for(i=0;i<model->nr_class-1;i++) 
    free(model->sv_coef[i]);
  /*
    if(model->param.svm_type == SVM_PLUS) {
    for(i=0;i<model->nr_class-1;i++) 
    free(model->sv_coef[i]);
    free(model->rho_star);
    free(model->SV_star);
    free(model->sv_coef_star);
    free(model->nSV_star);
    }
  
  free(model->SV);
  // free(model->sv_coef);
  free(model->rho);
  free(model->label);
  free(model->probA);
  free(model->probB);
  free(model->nSV);
free(model);

}
*/
