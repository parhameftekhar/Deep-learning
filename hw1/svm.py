import numpy as np


class SVM:
    def __init__(self, n_features: int, n_classes: int, std: float):
        """
        n_features: number of features in (or the dimension of) each instance
        n_classes: number of classes in the classification task
        std: standard deviation used in the initialization of the weights of svm
        """
        self.n_features, self.n_classes = n_features, n_classes
        self.cache = None
        ################################################################################
        # TODO: Initialize the weights of svm using random normal distribution with    #
        # standard deviation equals to std.                                            #
        ################################################################################
        #self.W = np.random.normal(0, std, (self.n_features + 1, self.n_classes))
        self.W = np.random.normal(0, std, (self.n_features, self.n_classes))

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def loss(self, X: np.ndarray, y: np.ndarray, reg_coeff: float):
        """
        X: training instances as a 2d-array with shape (num_train, n_features)
        y: labels corresponsing to the given training instances as a 1d-array with shape (num_train,)
        reg_coeff: L2-regularization coefficient
        """
        loss = 0.0
        ################################################################################
        # TODO: Compute the hinge loss specified in the notebook and save it in the    #                                                   # loss variable.                                                               #
        # NOTE: YOU ARE NOT ALLOWED TO USE FOR LOOPS!                                  #
        # Don't forget L2-regularization term in your implementation!                  #
        # You might need some values computed here when you will update the weights.   #
        # save them in self.cache attribute and use them in update_weights method.     #
        ################################################################################
        N, d = X.shape
        # zarbe vaznha dar bordarhaye vijegi
        All_s = np.matmul(self.W.T, X.T)
        # bedast avardane bordare emtiaz class dorost
        All_sy = All_s[y, range(N)]
        # ezafe kardan margin 1 va maximum giri
        Relu = np.maximum(All_s - All_sy + 1, 0)
        # be dast avardane Li ke hamun loss baraye har vurudi ast va kam kardan 1 bekhatere halati ke j = y(i) mishavad va nabayad dar loss hesab shavad
        All_L = np.sum(Relu, axis=0) - 1
        # be dast avardan loss kol
        loss = 1/N*(np.sum(All_L)) + reg_coeff * (np.sum(self.W*self.W))

        ####### in ghesmat vase gradian hesab kardane

        # mask baraye maghadire mosbate matrise emtiaze ha va az boolean be binary tabdil kardan anha
        mask = ((All_s - All_sy + 1) > 0).astype(int)
        # be dast avardan tedad bari ke -w_y(i) kam mishe az emtiaze class haye dg vase inpute i om va baz ham kam kardane 1 vase halati ke j = y(i) ast
        num_col_mask = -(np.sum(mask, axis=0) - 1)
        # ja gozarie maghadir be dast amade bala dar jaygahe motenazere class doroste har input dar matrise mask
        mask[y, range(N)] = num_col_mask
        # mohasebeye gradian 
        self.cache = (1/N)*np.matmul(X.T, mask.T) + 2*reg_coeff*self.W

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return loss
        
    def update_weights(self, learning_rate: float):
        """
        Updates the weights of the svm using the gradient of computed loss w.r.t. the weights. 
        learning_rate: learning rate that will be used in gradient descent to update the weights
        """
        grad_W = self.cache
        ################################################################################
        # TODO: Compute the gradient of loss computed above w.r.t the svm weights.     # 
        # the gradient will be used for updating the weights.                          #
        # NOTE: YOU ARE NOT ALLOWED TO USE FOR LOOPS!                                  #
        # Don't forget L2-regularization term in your implementation!                  #
        # You can use the values saved in cache attribute previously during the        #
        # computation of loss here.                                                    # 
        ################################################################################

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        self.W -= learning_rate * grad_W
        return grad_W
        
    def predict(self, X):
        """
        X: Numpy 2d-array of instances
        """
        y_pred = None
        ################################################################################
        # TODO: predict the labels for the instances in X and save them in y_pred.     #
        # Hint: You might want to use np.argmax.                                       #
        ################################################################################
        y_pred = np.argmax(np.matmul(self.W.T,X.T), axis=0)
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return y_pred