#include <stdio.h>
#include "lbfgs.h"

static lbfgsfloatval_t evaluate0(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    lbfgsfloatval_t fx = 0.0;

    for (i = 0;i < n;i += 2) {
        lbfgsfloatval_t x1 = 1.0 - x[i];
        lbfgsfloatval_t x2 = 10.0 * (x[i+1] - x[i] * x[i]);
        g[i+1] = 20.0 * x2;
        g[i] = -2.0 + 2.0 * x[i] - 200.0 * ((x[i+1] - x[i] * x[i])) * 2.0 * x[i];
        fx += x1 * x1 + x2 * x2;
    }
    return fx;
}

static lbfgsfloatval_t evaluate1(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    lbfgsfloatval_t fx = 0.0;
    lbfgsfloatval_t x1 = x[0];
    lbfgsfloatval_t x2 = x[1];
    g[0] = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
    g[1] = 200 * (x2 - x1 * x1);
    fx = 100 * (x2 - x1 * x1) * (x2 - x1 * x1) + (1 - x1) * (1 - x1);
    return fx;
}

static lbfgsfloatval_t evaluate2(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    lbfgsfloatval_t fx = 0.0;
    lbfgsfloatval_t x1 = x[0];
    lbfgsfloatval_t x2 = x[1];
    g[0] = 2*(x1 + 2*x2 - 7.0) + 2*(2*x1 + x2 - 5.0)*2;
    g[1] = 2*(x1 + 2*x2 - 7.0)*2 + 2*(2*x1 + x2 - 5.0);
    fx = (x1 + 2*x2 - 7.0) * (x1 + 2*x2 - 7.0) + (2*x1 + x2 - 5.0) * (2*x1 + x2 - 5.0);
    return fx;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

#define N   100

float c_lbfgs_0()
{
    int i, ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N);
    lbfgs_parameter_t param;

    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return -1;
    }

    /* Initialize the variables. */
    for (i = 0;i < N;i += 2) {
        x[i] = -1.2;
        x[i+1] = 1.0;
    }

    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_linesearch = 20;
    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */
    ret = lbfgs(N, x, &fx, evaluate0, NULL, NULL, &param);

    // /* Report the result. */
    // printf("L-BFGS optimization terminated with status code = %d\n", ret);
    // printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);

    lbfgs_free(x);
    return fx;
}

float c_lbfgs_1()
{
    int i, ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(2);
    lbfgs_parameter_t param;

    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return -1;
    }

    /* Initialize the variables. */
    x[0] = -1.2;
    x[1] = 1.0;   
    
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_linesearch = 20;
    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */
    ret = lbfgs(2, x, &fx, evaluate1, NULL, NULL, &param);

    // /* Report the result. */
    // printf("L-BFGS optimization terminated with status code = %d\n", ret);
    // printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);

    lbfgs_free(x);
    return fx;
}

float c_lbfgs_2()
{
    int i, ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(2);
    lbfgs_parameter_t param;

    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return -1;
    }

    /* Initialize the variables. */
    x[0] = -1.2;
    x[1] = 1.0;   
    
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_linesearch = 20;
    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */
    ret = lbfgs(2, x, &fx, evaluate2, NULL, NULL, &param);

    // /* Report the result. */
    // printf("L-BFGS optimization terminated with status code = %d\n", ret);
    // printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);

    lbfgs_free(x);
    return fx;
}