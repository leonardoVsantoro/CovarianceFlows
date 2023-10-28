import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from numpy.random import multivariate_normal
from scipy.ndimage import rotate
from tqdm import tqdm

def curve(cov):
    return multivariate_normal(np.zeros(cov.shape[0]),cov)

def plotFlow_3d(X,N):
    cmap = matplotlib.cm.coolwarm
    vmin= X.min(); vmax= X.max()
    time_grid = np.linspace(0,1,N)
    d = X.shape[1]
    xx, yy = np.meshgrid(np.arange(d),np.arange(d) )
    AZIM= 190; ELEV = 10; DIST=10
    fig = plt.figure(figsize = (20,5))
    fig.suptitle('Brownian Motion to Brownian Bridge', fontsize = 22)
    for i in range(N):
        ax = fig.add_subplot(1, N, i+1, projection='3d')
        ax.azim = AZIM;  ax.dist = DIST; ax.elev = ELEV; ax.set_zlim3d(-0.15, 1.15); ax.set_axis_off()
        surf2 = ax.plot_surface(xx, yy, X[(i*len(X))//N], cmap=cmap, linewidth=.3, antialiased=True, alpha = 1,vmin=vmin, vmax=vmax)
    axs = fig.axes;     
    for i,ax in enumerate(axs):
        axs[i].set_title('$t = {:.2f}$'.format(time_grid[i]), fontsize = 18, y=.85)
        ax.set_zlim(X.min(), X.max())
    axs[0].set_title('Brownian Motion\n $t = 0$', fontsize=18, y=.85)
    axs[-1].set_title('Brownian Bridge \n $t = 1$', fontsize=18,y=.85)  
    return fig
    
def plotflow_GP(X,N):
    cmap = matplotlib.cm.coolwarm
    vmin= X.min(); vmax= X.max()
    Dgrid = np.linspace(0,1,X.shape[1])
    
    fig,axs = plt.subplots(figsize = (18,3), ncols = N, sharey=True )
    fig.suptitle('corresponding gaussian processes', fontsize=16)
    for j, ax in enumerate(axs):
        curves = [curve(X[(j*len(X))//N]) for i in range(40)]
        for _ in curves:
            ax.plot(Dgrid, _, alpha=.75, lw=1)
        ax.set_axis_off()
    return fig

def plotFlow_3d_GP(X,N):
    cmap = matplotlib.cm.coolwarm
    vmin= X.min(); vmax= X.max()
    time_grid = np.linspace(0,1,N)
    d = X.shape[1]
    xx, yy = np.meshgrid(np.arange(d),np.arange(d) )
    AZIM= 190; ELEV = 10; DIST=10
    fig = plt.figure(figsize = (20,8))
    fig.suptitle('Brownian Motion to Brownian Bridge', fontsize = 22)
    for i in range(N):
        ax = fig.add_subplot(2, N, i+1, projection='3d')
        ax.azim = AZIM;  ax.dist = DIST; ax.elev = ELEV; ax.set_zlim3d(-0.15, 1.15); ax.set_axis_off()
        surf2 = ax.plot_surface(xx, yy, X[(i*len(X))//N], cmap=cmap, linewidth=.3, antialiased=True, alpha = 1,vmin=vmin, vmax=vmax)
        ax.set_title('$t = {:.2f}$'.format(time_grid[i]), fontsize = 18, y=.85)
        ax.set_zlim(X.min(), X.max())
        if i == 0:
            ax.set_title('Brownian Motion\n $t = 0$', fontsize=18, y=.85)
        if i == N-1:
            ax.set_title('Brownian Bridge \n $t = 1$', fontsize=18,y=.85)  

    vmin= X.min(); vmax= X.max()
    Dgrid = np.linspace(0,1,X.shape[1])

    for i in range(N):
        ax = fig.add_subplot(2, N, N+i+1)
        if i == N//2:
            ax.set_title('corresponding gaussian processes', fontsize=16)
        curves = [curve(X[i]) for _ in range(40)]
        for _ in curves:
            ax.plot(Dgrid, _, alpha=.75, lw=1)
        ax.set_axis_off()
    return fig

#cmap = matplotlib.cm.coolwarm
cmap = None

def plot_PCS(PCscores, n_components,n_clusters):
    colors = ['b','r', 'g' ,'c','y','m']
    markers = ['.','1','p','x','d']
    c = np.array([ [colors[j] for i in range(n//n_clusters)] for j in range(n_clusters)] ).ravel() 
    m = np.array([ [markers[j] for i in range(n//n_clusters)] for j in range(n_clusters)] ).ravel() 
    
    if n_components <= 2:
        fig,ax=plt.subplots(figsize = (12,6))
        ax.scatter(PCscores[0],PCscores[1], c=c,alpha=.7)
        _val_x = 1.1*np.abs(PCscores[0]).max(); _val_y = 1.1*np.abs(PCscores[1]).max()
        xmin, xmax, ymin, ymax = -_val_x, _val_x, -_val_y, _val_y; ticks_frequency = 1

        ax.spines['bottom'].set_position('zero'); ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        arrow_fmt = dict(markersize=4, color='black', clip_on=False)
        ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
        ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)
        
    if n_components > 2:
        fig = plt.figure(figsize=(12, 8))

        ax = fig.add_subplot(projection='3d')
        ax.scatter(PCscores[0],PCscores[1],PCscores[2], c=c)

        _val_x = 1.3*np.abs(PCscores[0]).max();
        _val_y = 1.3*np.abs(PCscores[1]).max();
        _val_z = 1.3*np.abs(PCscores[2]).max()

        arw = Arrow3D([0,0],[0,0],[-_val_z,_val_z], 
                      arrowstyle="->", color="k", lw = 1, mutation_scale=25); ax.add_artist(arw)
        arw = Arrow3D([0,0],[-_val_y,_val_y],[0,0], 
                      arrowstyle="->", color="k", lw = 1, mutation_scale=25); ax.add_artist(arw)
        arw = Arrow3D([-_val_x,_val_x],[0,0],[0,0], 
                      arrowstyle="->", color="k", lw = 1, mutation_scale=25); ax.add_artist(arw)
        ax.set_axis_off() ;ax.set_xlim(-_val_x,_val_x);ax.set_ylim(-_val_y,_val_y);ax.set_xlim(-_val_z,_val_z)
        
    
    fig.suptitle('Principal Component Scores', fontsize = 20)
    plt.show()


def plotOneFlow_heatmaps(X ,N,cmap=cmap, vmin =None, vmax = None, suptitle = None, title = None, suptitle_size= 21, title_size=18, xtitle = -.3):
    
    fig, axs = plt.subplots(figsize = (18,3 ), ncols = N )
    if suptitle is not None:
        fig.suptitle('{}'.format(suptitle), fontsize =suptitle_size )
    if vmin is None:
        vmin = np.percentile(X.ravel(),.5); 
    if vmax is None:
        vmax = np.percentile(X.ravel(),99.5);
    if title is not None:
        axs[0].set_title('${}$'.format(title),fontsize=title_size, x = xtitle,y=0.45)
    for j, ax in enumerate(axs):
        sns.heatmap(X[(j*len(X))//N], square = True, vmin = vmin, vmax = vmax, cbar=False, ax=ax, cmap=cmap); 
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_xlabel('t = {:.1f}'.format(j/N))
        ax.tick_params(left=False, bottom=False); 
    return fig

def plotFlow_heatmaps(Xs , tgrid_plot,cmap=cmap, suptitles = None, titles = None,  vmin_ =None, vmax_ = None,
                      suptitle_size= 21, title_size=18, xtitle = -.45):
    fig, _axs = plt.subplots(figsize = (16,3*len(Xs) ), ncols = tgrid_plot, nrows = len(Xs) ); _axs = _axs.reshape(-1,tgrid_plot)
    for (axs, X,title,suptitle) in tqdm(zip(_axs, Xs,titles,suptitles), total = len(Xs)):
        if suptitle is not None:
            axs[tgrid_plot//2].set_title('{}'.format(suptitle), fontsize =suptitle_size )
        if vmin_ is None:
            vmin = np.percentile(X.ravel(),.5); vmax = np.percentile(X.ravel(),99.5);
        else:
            vmin = vmin_; vmax = vmax_;
            
        if title is not None:
            axs[0].set_title('${}$'.format(title),fontsize=title_size, x = xtitle,y=0.45)
        for j, ax in enumerate(axs):
            sns.heatmap(X[(j*len(X))//tgrid_plot], square = True, vmin = vmin, vmax = vmax, cbar=False, ax=ax, cmap=cmap); 
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_xlabel('t = {:.1f}'.format(j/tgrid_plot))
            ax.tick_params(left=False, bottom=False); 
    fig.subplots_adjust(hspace=0.4); # fig.subplots_adjust(wspace=0.1)
    return fig
