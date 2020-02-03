import numpy as np
import scipy.io as io

from yaocptool.esn import sparsity, grad_tanh


class RecurrentNeuralNetworkOptions:
    def __init__(self,
                 gamma=0.5, ro=1.0, psi=0.5, in_scale=0.1,
                 bias_scale=0.5, alpha=10., forget=1.):
        self.forget = forget
        self.alpha = alpha
        self.bias_scale = bias_scale
        self.in_scale = in_scale
        self.psi = psi
        self.ro = ro
        self.gamma = gamma


class RecurrentNeuralNetwork:
    def __init__(self, neu, n_in, n_out,
                 gamma=0.5, ro=1.0, psi=0.5, in_scale=0.1,
                 bias_scale=0.5, alpha=10., forget=1., out_scale=0,
                 initial_filename="initial", load_initial=False, save_initial=False,
                 output_feedback=False,
                 noise_amplitude=0):
        """
        Recurrent Neural Network

            x(k+1) = gamma *

        :param neu: Number of neurons in the reservoir of the ESN
        :param n_in: Number of inputs to the ESN.
        :param n_out: Number of outputs in the ESN.
        :param gamma: The leak rate of the ESN.
        :param ro: Largest eigenvalue of the reservoir matrix.
        :param psi: sparseness of the reservoir matrix, don't worry about this parameter and keep it 0.0,
        :param in_scale: scaling of input weights related to the reservoir weights.
        :param bias_scale: scaling of the bias weights related to the reservoir weights.
        :param alpha: parameter for P initialization when applying RLS. P_0 = (1/alpha)*I.
        :param forget: Forgetting factor of the RLS.
        :param initial_filename: Name of the file where the initial conditions shall be stored.
        :param load_initial: Will the initial conditions be loaded from a file?
        :param save_initial: Will the initial conditions be saved into a file?
        :param output_feedback: Will there be output feedback?
        :param noise_amplitude: Amplitude of the noise applied inside the reservoir. For regularization.
        :param out_scale: Scaling of the feedback weights in relation to the reservoir weights.
        """
        # All matrices are initialized under the normal distribution.
        self.n_in = n_in
        self.n_out = n_out
        self.neu = neu

        self.psi = psi  # the network's sparsity, in 0 to 1 notation

        self.W_rr0, self.W_ir0, self.W_br0, self.W_or0 = self.create_initial_weights()

        # reservoir-output weight matrix
        self.W_ro = np.random.normal(0, 1, [n_out, neu + 1])
        self.leakrate = gamma  # the network's leak rate
        self.ro = ro  # the network's desired spectral radius
        self.in_scale = in_scale  # the scaling of W_ir.
        self.bias_scale = bias_scale  # the scaling of W_br

        # learning rate of the Recursive Least Squares Algorithm
        self.alpha = alpha
        # forget factor of the RLS Algorithm
        self.forget = forget
        self.output_feedback = output_feedback

        self.a = np.zeros([neu, 1], dtype=np.float64)
        # save if save is enabled
        if save_initial:
            self.save_initial_fun(initial_filename)

        # load if load is enabled
        if load_initial:
            self.load_initial(initial_filename)

        # the probability of a member of the Matrix W_rr being zero is psi.
        self.W_rr = self.W_rr0
        # forcing W_rr to have ro as the maximum eigenvalue
        eigs = np.linalg.eigvals(self.W_rr)
        radius = np.abs(np.max(eigs))

        # normalize matrix
        self.W_rr = self.W_rr / radius
        # set its spectral radius to rho
        self.W_rr *= ro

        # scale tbe matrices
        self.W_br = bias_scale * self.W_br0
        self.W_ir = in_scale * self.W_ir0
        self.W_or = out_scale * self.W_or0
        self.noise = noise_amplitude

        # initial conditions variable forget factor.
        self.sigma_e = 0.001 * np.ones(n_out)
        self.sigma_q = 0.001
        self.sigma_v = 0.001 * np.ones(n_out)
        self.K_a = 6
        self.K_b = 3 * self.K_a

        # covariance matrix
        self.P = np.eye(neu + 1) / alpha

    def create_initial_weights(self):
        # Reservoir Weight matrix.
        w_rr0 = np.random.normal(0, 1, [self.neu, self.neu])
        w_rr0 = sparsity(w_rr0, self.psi)

        # input-reservoir weight matrix
        w_ir0 = np.random.normal(0, 1, [self.neu, self.n_in])
        # bias-reservoir weight matrix
        w_br0 = np.random.normal(0, 1, [self.neu, 1])
        w_or0 = np.random.normal(0, 1, [self.neu, self.n_out])

        return w_rr0, w_ir0, w_br0, w_or0

    def training_error(self, ref):
        """
            Compute current training error between outputs and a reference vector

        :param ref: reference
        :return: error
        """
        ref = np.array(ref, dtype=np.float64)
        if self.n_out > 1:
            ref = ref.reshape(len(ref), 1)

        e = np.dot(self.W_ro, np.vstack((np.atleast_2d(1.0), self.a))) - ref
        return e

    def train(self, ref):
        """


        :param ref: vector of all output references in a given time.
        """
        e = self.training_error(ref)
        # the P equation step by step
        a_wbias = np.vstack((np.atleast_2d(1.0), self.a))
        self.P = self.P / self.forget - np.dot(np.dot(np.dot(self.P, a_wbias), a_wbias.T), self.P) / (
                self.forget * (self.forget + np.dot(np.dot(a_wbias.T, self.P), a_wbias)))

        for output in range(self.n_out):
            # Transpose respective output view..
            theta = self.W_ro[output, :]
            theta = theta.reshape([self.neu + 1, 1])

            # error calculation
            theta = theta - e[output] * np.dot(self.P, a_wbias)
            theta = theta.reshape([1, self.neu + 1])

            self.W_ro[output, :] = theta

    def update(self, inp, y_in=np.atleast_2d(0)):
        """
        Updates the network

        :param np.array inp: has to have same size as n_in
        :param y_in:
        :return: Returns the output as shape (2,1), so if you want to plot the data, a buffer is mandatory.
        :rtype: np.array
        """

        input_ = np.array(inp)
        input_ = input_.reshape(input_.size, 1)
        y_in = np.array(y_in)
        y_in = y_in.reshape(y_in.size, 1)
        if (y_in == 0).all():
            y_in = np.zeros([self.n_out, 1])
        if input_.size == self.n_in:
            z = np.dot(self.W_rr, self.a) + np.dot(self.W_ir, input_) + self.W_br
            if self.output_feedback:
                z += np.dot(self.W_or, y_in)
            if self.noise > 0:
                z += np.random.normal(0, self.noise, [self.neu, 1])
            self.a = (1 - self.leakrate) * self.a + self.leakrate * np.tanh(z)

            a_wbias = np.vstack((np.atleast_2d(1.0), self.a))
            y = np.dot(self.W_ro, a_wbias)
            return y
        else:
            raise ValueError("input must have size n_in")

    def train_lms(self, ref):
        learning_rate = 1

        e = self.training_error(ref)
        for out in range(self.n_out):
            theta = self.W_ro[out, :]
            theta = theta.reshape([self.neu, 1])
            theta = theta - learning_rate * e * self.a / np.dot(self.a.T, self.a)
            self.W_ro[out, :] = theta.T

    def offline_training(self, x_matrix, y_vector, regularization, warmupdrop):
        """
        X is a matrix in which X[row,:] is all parameters at time row. Y is a vector of desired outputs.
        """
        a_matrix = np.empty([y_vector.shape[0] - warmupdrop, self.neu])
        for row in range(y_vector.shape[0]):
            if self.output_feedback:
                if row > 0:
                    self.update(x_matrix[row, :], y_vector[row - 1, :])
                else:
                    self.update(x_matrix[row, :])
                if row > warmupdrop:
                    a_matrix[row - warmupdrop, :] = self.a.T
            else:
                self.update(x_matrix[row, :])
                if row > warmupdrop:
                    a_matrix[row - warmupdrop, :] = self.a.T

        a_wbias = np.hstack((np.ones([a_matrix.shape[0], 1]), a_matrix))

        p_matrix = np.dot(a_wbias.T, a_wbias)
        r_matrix = np.dot(a_wbias.T, y_vector[warmupdrop:])

        theta = np.linalg.solve(p_matrix + regularization * np.eye(self.neu + 1, self.neu + 1), r_matrix)
        self.W_ro = theta.T

    def reset(self):
        self.a = np.zeros([self.neu, 1])

    def get_forgetingfactor(self):
        return self.forget

    def get_state_from(self, network):
        self.a = network.a

    def get_derivative_df_du(self, inp, y_in=np.atleast_2d(0)):
        input_ = np.array(inp)
        input_ = input_.reshape(input_.size, 1)

        z = np.dot(self.W_rr, self.a) + np.dot(self.W_ir, input_) + self.W_br
        if not (y_in == 0).all():
            y_in = np.array(y_in)
            y_in = y_in.reshape(y_in.size, 1)
            z += np.dot(self.W_or, y_in)

        jacobian = grad_tanh(z)

        return np.dot(jacobian, self.W_ir)

    def get_derivative_df_dx(self, inp, y_in=np.atleast_2d(0)):
        input_ = np.array(inp)
        input_ = input_.reshape(input_.size, 1)

        z = np.dot(self.W_rr, self.a) + np.dot(self.W_ir, input_) + self.W_br
        if not (y_in == 0).all():
            y_in = np.array(y_in)
            y_in = y_in.reshape(y_in.size, 1)
            z += np.dot(self.W_or, y_in)

        jacobian = grad_tanh(z)

        z1 = self.W_rr
        if self.output_feedback:
            z1 += np.dot(self.W_or, self.W_ro[:, 1:])
        z1 = np.dot(jacobian, z1)

        return (1 - self.leakrate) * np.eye(self.neu) + self.leakrate * z1

    def get_current_output(self):
        a_wbias = np.vstack((np.atleast_2d(1.0), self.a))
        return np.dot(self.W_ro, a_wbias)

    def covariance_reset(self, diag_alpha):
        self.P = np.eye(self.neu) / diag_alpha

    def copy_weights(self, network):
        if self.W_ro.shape == network.Wro.shape:
            self.W_ro = np.copy(network.Wro)
            self.W_rr = np.copy(network.Wrr)
            self.W_ir = np.copy(network.Wir)
            self.W_br = np.copy(network.Wbr)
            self.W_br = np.copy(network.Wbr)
        else:
            raise ValueError("shapes of the weights are not equal")

    def save_reservoir(self, file_name):
        data = {'W_rr': self.W_rr,
                'W_ir': self.W_ir,
                'W_br': self.W_br,
                'W_ro': self.W_ro,
                'a0': self.a}
        io.savemat(file_name, data)

    def load_reservoir(self, file_name):
        data = {}
        io.loadmat(file_name, data)
        self.load_reservoir_from_array(data)

    def load_reservoir_from_array(self, data):
        self.W_rr = data['W_rr']
        self.W_ir = data['W_ir']
        self.W_br = data['W_br']
        self.W_ro = data['W_ro']

        if 'a0' in data:  # check by Eric
            self.a = data['a0']

        # added by Eric - start
        if 'Wro_b' in data:
            self.W_ro = np.hstack((data['Wro_b'], self.W_ro))

        if 'leak_rate' in data:
            try:
                self.leakrate = data['leak_rate'][0][0]
            except TypeError:
                self.leakrate = data['leak_rate']

        self.neu = self.W_rr.shape[0]
        self.n_in = self.W_ir.shape[1]
        self.n_out = self.W_ro.shape[0]
        # added by Eric - end

    def load_initial(self, filename):
        data = {}

        io.loadmat(filename, data)
        self.W_rr0 = data['W_rr']
        self.W_ir0 = data['W_ir']
        self.W_br0 = data['W_br']
        self.W_ro = data['W_ro']
        # self.W_or0 = data['W_or']
        self.a = data['a0']

    def save_initial_fun(self, filename):
        data = {'W_rr': self.W_rr0, 'W_ir': self.W_ir0, 'W_br': self.W_br0, 'W_or': self.W_or0, 'W_ro': self.W_ro,
                'a0': self.a}
        print("saving reservoir")
        io.savemat(filename, data)

    @staticmethod
    def new_rnn_from_weights(weights):
        esn_4tanks = RecurrentNeuralNetwork(
            neu=400,
            n_in=2,
            n_out=2,
            gamma=0.1,
            ro=0.99,
            psi=0.0,
            in_scale=0.1,
            bias_scale=0.1,
            initial_filename="4tanks1",
            load_initial=False,
            save_initial=False,
            output_feedback=False)
        esn_4tanks.load_reservoir_from_array(weights)
        esn_4tanks.reset()
        return esn_4tanks
