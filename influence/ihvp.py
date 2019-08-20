from scipy.optimize import fmin_ncg
import numpy as np

def update_feed_dict_with_v_placeholder(trained_model, feed_dict, vec):
    for pl_block, vec_block in zip(trained_model.v_placeholder, vec):
        #print("pl_block", pl_block)
        #print("vec_block", vec_block)
        feed_dict[pl_block] = vec_block
    return feed_dict

def minibatch_hessian_vector_val(trained_model, v):

    num_examples = trained_model.num_train_examples
    if trained_model.mini_batch == True:
        batch_size = 100
        assert num_examples % batch_size == 0
    else:
        batch_size = trained_model.num_train_examples

    num_iter = int(num_examples / batch_size)

    trained_model.reset_datasets()
    hessian_vector_val = None
    for i in range(num_iter):
        feed_dict = trained_model.fill_feed_dict_with_batch(trained_model.data_sets.train, batch_size=batch_size)
        #print("batch", feed_dict)
        # Can optimize this
        feed_dict = update_feed_dict_with_v_placeholder(trained_model, feed_dict, v)
        #print("v", feed_dict)
        hessian_vector_val_temp = trained_model.sess.run(trained_model.hessian_vector, feed_dict=feed_dict)
        if hessian_vector_val is None:
            hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
        else:
            hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]

    hessian_vector_val = [a + trained_model.damping * b for (a,b) in zip(hessian_vector_val, v)]

    return hessian_vector_val

def get_fmin_loss_fn(trained_model, v):
    vec_to_list = get_vec_to_list_fn(trained_model)
    def get_fmin_loss(x):
        print("get_fmin_loss")
        print("x", x)
        hessian_vector_val = minibatch_hessian_vector_val(trained_model, vec_to_list(x))
        print("hvp", hessian_vector_val)
        # print(len(hessian_vector_val))
        # print(len(hessian_vector_val[0]))
        print("v", v)
        return_val = 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
        print("fmin_loss", return_val)
        return return_val

    return get_fmin_loss


def get_fmin_grad_fn(trained_model, v):
    vec_to_list = get_vec_to_list_fn(trained_model)
    def get_fmin_grad(x):
        print("get_fmin_grad")

        hessian_vector_val = minibatch_hessian_vector_val(trained_model, vec_to_list(x))
        return_val = np.concatenate(hessian_vector_val) - np.concatenate(v)
        print("x", x)
        print("hvp", hessian_vector_val)
        print("v", v)
        print("fmin_grad", return_val)
        return return_val

    return get_fmin_grad


def get_cg_callback(trained_model, v, verbose):
    fmin_loss_fn = get_fmin_loss_fn(trained_model, v)
    vec_to_list = get_vec_to_list_fn(trained_model)

    def fmin_loss_split(x):
        hessian_vector_val = minibatch_hessian_vector_val(trained_model, vec_to_list(x))

        return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

    def cg_callback(x):
        # x is current params
        v = trained_model.vec_to_list(x)
        idx_to_remove = 5

        single_train_feed_dict = trained_model.fill_feed_dict_with_one_ex(trained_model.data_sets.train, idx_to_remove)
        train_grad_loss_val = trained_model.sess.run(trained_model.grad_total_loss_op, feed_dict=single_train_feed_dict)
        predicted_loss_diff = np.dot(np.concatenate(v),
                                     np.concatenate(train_grad_loss_val)) / trained_model.num_train_examples

        if verbose:
            print('Function value: %s' % fmin_loss_fn(x))
            quad, lin = fmin_loss_split(x)
            print('Split function value: %s, %s' % (quad, lin))
            print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

    return cg_callback


def get_vec_to_list_fn(trained_model):
    params_val = trained_model.sess.run(trained_model.params)
    trained_model.num_params = len(np.concatenate(params_val))
    print('Total number of parameters: %s' % trained_model.num_params)

    def vec_to_list(v):
        return_list = []
        cur_pos = 0
        for p in params_val:
            return_list.append(v[cur_pos: cur_pos + len(p)])
            cur_pos += len(p)

        assert cur_pos == len(v)
        return return_list

    return vec_to_list


def get_inverse_hvp_cg(trained_model, v, verbose=True):
    fmin_loss_fn = get_fmin_loss_fn(trained_model, v)
    fmin_grad_fn = get_fmin_grad_fn(trained_model, v)
    cg_callback = get_cg_callback(trained_model, v, verbose)
    vec_to_list = get_vec_to_list_fn(trained_model)

    fmin_results = fmin_ncg(
        f=fmin_loss_fn,
        x0=np.concatenate(v),
        fprime=fmin_grad_fn,
        fhess_p=trained_model.get_fmin_hvp,
        callback=cg_callback,
        avextol=1e-8,
        maxiter=100)

    return vec_to_list(fmin_results)