import torch
import numpy as np
from scipy.spatial.distance import cdist
import random
from pyemd import emd
from matplotlib.colors import to_rgba
import matplotlib.animation as animation

def gaussian_kernel(x, x0, temperature=1e-1):
    dim = x0.size(1)
    x = x.view((1, -1))
    exp_term = torch.sum(- 0.5 * (x - x0) ** 2, dim=1)
    main_term = torch.exp(exp_term / (2 * temperature))
    coeff = 1. / torch.sqrt(torch.Tensor([2 * torch.pi * temperature])) ** dim
    prod = coeff * main_term
    return torch.sum(prod) / x0.size(0)

def get_likelihood(data, samples, temperature):
    lh = torch.zeros(samples.size(0))
    for i in range(samples.size(0)):
        lh[i] = gaussian_kernel(samples[i,:], data, temperature)
    return torch.mean(lh)

def get_nll(data, samples, temperature=1e-1):
    return -torch.log(get_likelihood(data, samples, temperature))

def get_emd(d1, d2):
    d_comb = np.concatenate((d1, d2), axis=0)
    dist = np.linalg.norm((d_comb), axis=1).reshape((-1,1))
    d1 = np.concatenate((np.zeros((d1.shape[0], 1)), d1), axis=1)
    d2 = np.concatenate((np.ones((d2.shape[0], 1)), d2), axis=1)
    d_comb = np.concatenate((d1, d2), axis=0)
    app = np.concatenate((dist, d_comb), axis=1)
    app = app[app[:, 0].argsort()]
    d1_sig, d2_sig = 1 - app[:, 1], app[:, 1]
    dist_sorted = app[:, 2:]
    dist = cdist(dist_sorted, dist_sorted)
    d1_sig = d1_sig.copy(order='C')
    d2_sig = d2_sig.copy(order='C')
    dist = dist.copy(order='C')
    return emd(d1_sig, d2_sig, dist)

def split_data(X, y, split_ratio=0.9):
    size = len(X)
    indices = torch.randperm(size)
    split_idx = int(split_ratio * size)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def sample(*args,size):
    leng_ = len(args[0])
    for x in args[1:]:
        leng = len(x)
        assert leng_ == leng

    if size>leng_:
        return args
    else:
        ind = np.random.choice(leng_, size, replace=False)
        
        sampled_args = ()
        for x in args:
            sampled_args = sampled_args + (x[ind],)
        if len(args)>1:           
            return sampled_args
        else:
            return sampled_args[0]

def seed_everything(seed):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)

@torch.no_grad()
def viz_clf(ax,model, DEVICE=torch.device('cpu'), hard=False, 
            bound_x=(-1,1), bound_y=(-1,1), step=0.001, pad=0.05, contour=False, **kwargs):

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # pad the bounds for better plots
    bound_x=(bound_x[0]-pad,bound_x[1]+pad)
    bound_y=(bound_y[0]-pad,bound_y[1]+pad)

    c0 = torch.Tensor(to_rgba('#440154FF')).to(DEVICE)
    c1 = torch.Tensor(to_rgba('#FDE725FF')).to(DEVICE)
    x1 = torch.arange(*bound_x, step=step, device=DEVICE)
    x2 = torch.arange(*bound_y, step=step, device=DEVICE)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='xy')  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    leng = model_inputs.shape[0]
    width = model_inputs.shape[1]

    # checks if the model is an instance of sklearn classifiers, not ideal
    sk_clf = "sklearn" in str(model.__class__)

    if sk_clf:
        if hard:# hax
            preds = torch.from_numpy(model.predict(model_inputs.reshape(-1,2)).reshape(leng,width,-1))
        else:
            preds = torch.from_numpy(model.predict_proba(model_inputs.reshape(-1,2))[:,1].reshape(leng,width,-1))
    else:
        model.to(DEVICE)
        model.eval()
        if hard:
            preds = 1.0*(model(model_inputs) > 0.5)
        else:
            preds = model(model_inputs)

    if contour:
        if  isinstance(xx1, torch.Tensor):
            xx1 = xx1.cpu().numpy()
            xx2 = xx2.cpu().numpy()
            preds = preds.cpu().numpy()
            preds = preds.squeeze()
        
        ax.contour(xx1, xx2, preds, levels=[0.5], colors='k', linestyles='--',**kwargs)
        ax.set_xlim(bound_x)
        ax.set_ylim(bound_y)
    else:
        output_image = (1 - preds) * c0[None,None] + preds * c1[None,None]  # Specifying "None" in a dimension creates a new one
        if  isinstance(output_image, torch.Tensor):
            output_image = output_image.cpu().numpy()
        ax.imshow(output_image, origin='lower', extent=(*bound_x,*bound_y), alpha=0.6)


def animateScatter2d(fig,ax,seq_of_scatter_data, title="", x_bounds=(-1, 1), y_bounds=(-1, 1)):
    scat = ax.scatter([], [], s=10) 
    ax.set_title(f"{title}")

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)

    # Animation functions
    def init():
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    def update(frame):
        samples = seq_of_scatter_data[frame]
        scat.set_offsets(samples)
        ax.set_title(f"{title} - Step {frame}")  # Optional: Show step number
        return (scat,)

    # Create animation
    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(seq_of_scatter_data),
        init_func=init,
        blit=True,
        interval=40  # milliseconds between frames
    )

    return ani


