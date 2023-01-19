from model_linear.lib import *
import inspect

class Linear_Model_LM:
    def PF_degree_2():
        # LinearRegression degree 2
        polynomial_features = PolynomialFeatures(degree=2,
                                         include_bias=False)
        linear_regression = LinearRegression(n_jobs=-1)
        return  'PF_degree_2', Pipeline([("polynomial_features", polynomial_features),
                                    ("linear_regression", linear_regression)])
    def LM_PF_degree_1():
        # model Ridge degree 1
        return 'LM_PF_degree_1' , make_pipeline(PolynomialFeatures(1), Ridge(alpha=1e-5))
    def LM_PF_degree_2():
        # model Ridge degree 2
        return 'LM_PF_degree_2' , make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-5))
    def LM_PF_degree_3():
        # model Ridge degree 3
        return 'LM_PF_degree_3' , make_pipeline(PolynomialFeatures(3), Ridge(alpha=1e-5))
    def LM_Ridge():
        # model ridge default
        # LM_Ridge 
        return 'LM_Ridge' , Ridge(alpha=0.0, random_state=0, normalize=True)
    def LM_RidgeCV():
        # model RidgeCV default
        # LM_RidgeCV
        return 'LM_RidgeCV', RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    def LM_Bayesian_Ridge():
            # model BayesianRidge
        # LM_Bayesian-Ridge

        model_BayesianRidge = BayesianRidge(
            tol=1e-6, fit_intercept=False, compute_score=True)
        init = [1, 1e-4]
        model_BayesianRidge.set_params(alpha_init=init[0], lambda_init=init[1])
        return 'LM_Bayesian-Ridge', model_BayesianRidge
    def LM_SGD():  
        # model SGDRegressor
        # LM_SGD 
        model_SGDRegressor = make_pipeline(StandardScaler(),
                                        SGDRegressor(max_iter=1000, tol=1e-3, random_state=True))
        return 'LM_SGD', model_SGDRegressor
    def LM_ARD():
        # model ARDRegression
        # LM_ARD
        model_ARDRegression = ARDRegression(compute_score=True)
        return 'LM_ARD', model_ARDRegression
    # def LM_MT_ElasticNet():
    #         # model MultiTask
    #     # LM_MT_ElasticNet
    #     model_MultiTask = MultiTaskElasticNet(alpha=0.1)
    #     return 'LM_MT_ElasticNet', model_MultiTask
    def LM_Lasso():
            # model LassoLarsCV
        # LM_Lasso
        model_LarsCV = LassoLarsCV(cv=5)
        return 'LM_Lasso', model_LarsCV
    def LM_Kernel_Ridge():
            # Model KernelRidge
        # LM_Kernel_Ridge
        param_grid_KernelRidge = {"alpha": [1e0, 1e1, 1e-1, 1e-2]}
        model_KernelRidge = GridSearchCV(
            KernelRidge(), param_grid=param_grid_KernelRidge)
        return 'LM_Kernel_Ridge', model_KernelRidge
    def LM_Huber():
            # model huber
        #LM_Huber
        huber = HuberRegressor(alpha=0.0, epsilon=1.35)
        return 'LM_Huber', huber
    def LM_KRR_RBF():
            # model KRR
        #LM_KRR_RBF
        model_KRR = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                                param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                            "gamma": np.logspace(-2, 2, 5)})
        return 'LM_KRR_RBF', model_KRR

class Gaussian_GPR:
    def GPR_DP_WK():
            # model gaussian karnel default
        # GPR_DP_WK
        gp_kernel_1 = DotProduct() + WhiteKernel()
        model_gaussian_kernel_default = GaussianProcessRegressor(kernel=gp_kernel_1)
        return 'GPR_DP_WK', model_gaussian_kernel_default
    def GPR_Kernel_WK():
            # model graussian kernel
        # GPR_Kernel_WK
        gp_kernel_2 = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
            + WhiteKernel(1e-1)
        model_gaussian_kernel_2 = GaussianProcessRegressor(kernel=gp_kernel_2)
        return 'GPR_Kernel_WK', model_gaussian_kernel_2
    def GPR_RBF_kernel_1():
            # model gaussian correction
        # GPR_RBF_kernel_1
        k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
        k2 = 2.4**2 * RBF(length_scale=90.0) \
            * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
        # medium term irregularity
        k3 = 0.66**2 \
            * RationalQuadratic(length_scale=1.2, alpha=0.78)
        k4 = 0.18**2 * RBF(length_scale=0.134) \
            + WhiteKernel(noise_level=0.19**2)  # noise terms
        kernel_gpml_1 = k1 + k2 + k3 + k4
        model_gaussian_GPR = GaussianProcessRegressor(kernel=kernel_gpml_1, alpha=0,
                                                    optimizer=None, normalize_y=True)
        return 'GPR_RBF_kernel_1', model_gaussian_GPR
    def GPR_RBF_kernel_2():
            # Model gaussian correction 2
        # GPR_RBF_kernel_2
        k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
        k2 = 2.0**2 * RBF(length_scale=100.0) \
            * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                            periodicity_bounds="fixed")  # seasonal component
        # medium term irregularities
        k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
        k4 = 0.1**2 * RBF(length_scale=0.1) \
            + WhiteKernel(noise_level=0.1**2,
                        noise_level_bounds=(1e-5, np.inf))  # noise terms
        kernel_2 = k1 + k2 + k3 + k4
        model_gaussian_GPR_2 = GaussianProcessRegressor(kernel=kernel_2, alpha=0.5,
                                                        normalize_y=True)
        return 'GPR_RBF_kernel_2',model_gaussian_GPR_2
    def GPR_RBF_kernel_3():
            # model gaussian correction 3
        # GPR_RBF_kernel_3
        kernel_3 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
        model_gaussian_RBF = GaussianProcessRegressor(kernel=kernel_3,
                                                    alpha=0.0)
        return 'GPR_RBF_kernel_3', model_gaussian_RBF
    def GPR_RBF_kernel_4():
        # model gaussion kernel
        # GPR_RBF_kernel_4
        kernel_4 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        model_gaussian_kernel_3 = GaussianProcessRegressor(
            kernel=kernel_4, n_restarts_optimizer=9)
        return 'GPR_RBF_kernel_4', model_gaussian_kernel_3
    def GPR_RBF_kernel_5():    
        # model gaussion kernel
        # GPR_RBF_kernel_5
        kernel_5 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                                nu=1.5)
        model_gaussian_kernel_5 = GaussianProcessRegressor(kernel=kernel_5)
        return 'GPR_RBF_kernel_5',model_gaussian_kernel_5

class SVM:
    def SVM_SVC():
        # model SVC
        #SVM_SVC
        model_SVC = make_pipeline(StandardScaler(),
                                LinearSVR(random_state=0, tol=1e-6))
        return 'SVM_SVC', model_SVC
    def SVM_SVR():
        # model SVR
        # SVM_SVR
        model_SVR = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                                param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                            "gamma": np.logspace(-2, 2, 5)})
        return 'SVM_SVR',model_SVR
class Tree:
    def tree():
        # model tree
        #TREE
        model_tree = DecisionTreeRegressor(random_state=0)
        # model_tree.fit(X, y, sample_weight=None, check_input=True,
        #                  X_idx_sorted='deprecated')
        return 'Tree',model_tree
def call_all_method(name_class):
    methods = {}
    for name in dir(name_class):
        if not name.startswith('_'):
            attr = getattr(name_class,name)
            methods[name] = attr    
    return methods

def main():
    list_class_model={Linear_Model_LM,Gaussian_GPR,Tree,SVM}
    for class_model in list_class_model:
        for name,method in call_all_method(class_model).items():
            #print("calling: {}".format(name))
            try:
                name_model,model=method()
                print(name_model)
            except:
                continue
    
if __name__=='__main__':
    main()
    