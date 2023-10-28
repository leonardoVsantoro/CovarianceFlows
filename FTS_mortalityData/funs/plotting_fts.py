# library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
from datetime import datetime
from numpy.linalg import norm
from tqdm import tqdm
from miscellaneous import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import os
mypath = './data'
files_in_mypath = os.listdir(mypath); 


# generate a dictionary that maps geoIDs to full names of regions
iso_dict = {}
for filename in files_in_mypath:
    try:
        filepath = mypath+'/'+filename
        with open(filepath) as fp:
            text = fp.readline()
        i=0
        while text[i] != ',':
            i=i+1
        name = text[:i]
        if name[-1]==' ':
            name = name[:-1]
        iso_code = filepath[7:-11]
        iso_dict.update({iso_code:name})
    except:
    	None
# shorten USA dict to 'United States' -- for visualisation
iso_dict.update({'USA': 'United States'})





# ---- funs for plotting

def plot_clustering_onMap(kmeans,data,geoIDs,ax):
    labels =  kmeans.labels_
    # ##########
    cs = np.array(['' for i in labels])
    col_ids = ['RUS', 'USA', 'POL','LTU','NOR']
    cols = ['r','b','m','c','y']
    for _id, _c, _ in zip( col_ids, cols, np.unique(labels) ):
        cs[labels == labels[np.where(geoIDs ==_id)]] = _c     
    if '' in cs:
        cs = np.array(['' for i in labels])
        cols = ['r','b','m','c','y']
        for _lab, _c, _ in zip( np.unique(labels), cols, np.unique(labels) ):
            cs[labels == _lab] = _c     
    # ##########    
    sorted_labels = labels[np.argsort(labels)]
    sorted_IDs = np.array(geoIDs)[np.argsort(labels)]
    sorted_cs = cs[np.argsort(labels)]
    n = data.shape[0]

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres')).to_crs("EPSG:3395")
    world.loc[world.name == 'United States of America','name'] = 'United States'

    world = world[(world.name!="Antarctica")]; 
    for i, ID in enumerate(sorted_IDs):
        if world[world.name == iso_dict[ID]].size>0:
            world[world.name == iso_dict[ID]].plot(color=sorted_cs[i],ax=ax,alpha = .7)
        else:
            None
    world.boundary.plot(ax=ax, lw=.3,color='k')
    world[world.name == 'Germany'].plot(color=sorted_cs[sorted_IDs=='DEUTE'],ax=ax,alpha = .7)
    ax.set_title('K = {}'.format(np.unique(labels).size),fontsize = 18)
    ax.axis('off')


def plot_SimilarityMatrix(kmeans, geoIDs, data, xb = 10, yb = 8,yleg = 1):
    labels =  kmeans.labels_
    # ##########
    cs = np.array(['' for i in labels])
    col_ids = ['RUS', 'USA', 'POL','LTU','NOR']
    cols = ['r','b','m','c','y']
    for _id, _c, _ in zip( col_ids, cols, np.unique(labels) ):
        cs[labels == labels[np.where(geoIDs==_id)]] = _c     
    if '' in cs:
        cs = np.array(['' for i in labels])
        cols = ['r','b','m','c','y']
        for _lab, _c, _ in zip( np.unique(labels), cols, np.unique(labels) ):
            cs[labels == _lab] = _c     
    # ##########   
    sorted_labels = labels[np.argsort(labels)]
    sorted_IDs = np.array(geoIDs)[np.argsort(labels)]
    sorted_fullnames = [iso_dict[ID] for ID in sorted_IDs]
    sorted_data = data[np.argsort(labels)]
    sorted_cs = cs[np.argsort(labels)]
    n = data.shape[0]
    pairwise_dist_matrix = np.zeros(shape = (n,n))
    for i in range(n):
        for j in range(n):
            pairwise_dist_matrix[i,j] = norm(sorted_data[i] - sorted_data[j])
    pairwise_dist_matrix = (1 - pairwise_dist_matrix/pairwise_dist_matrix.max())


    fig, axs = plt.subplots(figsize = (20,8), ncols=2)
    ax=axs[0]
    ax.set_title('Similarity Matrix',fontsize=18)
    sns.heatmap(pairwise_dist_matrix,ax=ax,square=True,cbar=False)
    ax.set_xticklabels(sorted_fullnames,rotation = 90,fontsize = 13)
    ax.set_yticklabels(sorted_fullnames,rotation = 0,fontsize = 13)

    # -----
    ax=axs[1]
    ax.set_title('Class Inclusion Matrix',fontsize=18)
    
    for i in set(sorted_labels):
        _where = np.where(sorted_labels == i)[0]; cluster_size = len(_where)
        c = sorted_cs[_where][0]
        ax.fill_between(np.arange(_where[0],_where[-1]+2),
                        y1 =  n-np.ones(cluster_size+1)*_where[-1]-1, y2= n-np.ones(cluster_size+1)*_where[0]
                       , color = c)
    ax.set_xticks(np.arange(n)+.5); ax.set_yticks(np.arange(n)+.5);
    ax.set_xticklabels(sorted_fullnames,rotation = 90,fontsize = 13)
    ax.set_yticklabels(sorted_fullnames[::-1],rotation = 0,fontsize = 13)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig




def plot_Trends(kmeans, Xs, geoIDs, years, xb = 15, yb = .22,yleg = -1.1, xleg=-.8):
    labels =  kmeans.labels_
    # ##########
    cs = np.array(['' for i in labels])
    col_ids = ['RUS', 'USA', 'POL','LTU','NOR']
    cols = ['r','b','m','c','y']
    for _id, _c, _ in zip( col_ids, cols, np.unique(labels) ):
        cs[labels == labels[np.where(geoIDs==_id)]] = _c     
    if '' in cs:
        cs = np.array(['' for i in labels])
        cols = ['r','b','m','c','y']
        for _lab, _c, _ in zip( np.unique(labels), cols, np.unique(labels) ):
            cs[labels == _lab] = _c     
    # ##########      
    sorted_labels = labels[np.argsort(labels)]
    sorted_IDs = np.array(geoIDs)[np.argsort(labels)]
    sorted_cs = cs[np.argsort(labels)]
    sorted_Xs = Xs[np.argsort(labels)]

    
    N_clusters = len(set(labels))    
    fig,axs = plt.subplots(figsize = (20,5), ncols=N_clusters, nrows = 1, sharey=True); 
    fig.suptitle('Time Trends', fontsize = 20,y=1.12, x=.5); axs=axs.ravel()
    for i,ax in zip( np.arange(N_clusters), axs) :
        ax.set_title('Cluster {}'.format(i+1),fontsize =15)
        ixs = np.where(sorted_labels == i)[0]
        same_cluster_Xs = sorted_Xs[ixs]
        list_of_countries = [iso_dict[sorted_IDs[ix]] for ix in ixs]
        year_cmp = plt.cm.viridis(np.linspace(0,1,len(years)))
        for x,c,y in zip(same_cluster_Xs.mean(0),year_cmp,years):
            ax.plot(x, c=c,alpha=.8,label = y)
        ax.set_xlabel(' '.join(word for word in ['{} \n'.format(country) for country in list_of_countries]),
                     fontsize =12.5)
        
        ax.set_xticks(np.linspace(0,x.size,11))
        ax.set_xticklabels(np.linspace(0,100,11).astype(int), rotation = 45, fontsize = 11) 
#         ax.set_xlabel('age',fontsize = 12.5)
    axs[(N_clusters+1)//2].legend(loc='lower center',bbox_to_anchor=(xleg, yleg), ncol=13,fontsize = 12.5)
    axs[0].set_ylabel('log-mortality rate increments',fontsize = 12.5)

    a=r'$\bullet$'.split(); c = 'red'
    try:
        for i,ax in enumerate(axs):
            c = sorted_cs[np.where(sorted_labels == i)[0]][0]
            t = ax.text(xb, yb, a[0],color=c,fontsize=100)
    except:
        None
    return fig




def PlotElbow(data, n_clusters_list = np.arange(2,13), ninit=20):
    print('computing elbow...')
    # elbow method 
    inertia = []
    distortions = []
    
    
    for k in tqdm(n_clusters_list,ncols= 50):
        km = KMeans(n_clusters=k,n_init = ninit)
        km = km.fit(data)
        inertia.append(km.inertia_)
        distortions.append(sum(np.min(cdist(data,km.cluster_centers_, 'euclidean'),axis=1)) /data.shape[0])
        
    
    fig,axs = plt.subplots(figsize = (20,5), ncols = 2)
    fig.suptitle('Elbow method for optimal number of clusters', fontsize =19, y =1.05)
    
    ax = axs[0];  ax.set_title('Inertia',fontsize = 15)
    ax.plot(n_clusters_list, inertia, 'bx-');ax.set_xticks(n_clusters_list);
    ax.set_xlabel('number of clusters',fontsize = 13)
    ax.set_ylabel('score' ,fontsize = 13)
    kl = KneeLocator(n_clusters_list, inertia, curve="convex", direction="decreasing")
    ax.scatter(kl.elbow, inertia[kl.elbow-2], s=300,c='r',alpha=.3)
    ax.vlines(kl.elbow, min(inertia) , inertia[kl.elbow-2], lw=.4, color='r')
    
    
    ax = axs[1];  ax.set_title('Distortion',fontsize = 15)
    ax.plot(n_clusters_list, distortions, 'bx-');ax.set_xticks(n_clusters_list);
    ax.set_xlabel('number of clusters',fontsize = 13)
    ax.set_ylabel('score' ,fontsize = 13)
    kl = KneeLocator(n_clusters_list, distortions, curve="convex", direction="decreasing")
    ax.scatter(kl.elbow, distortions[kl.elbow-2], s=300,c='r',alpha=.3)
    ax.vlines(kl.elbow, min(distortions) , distortions[kl.elbow-2], lw=.4, color='r')
    
    plt.subplots_adjust(wspace=0.4,)
    
    return fig


def plotTwoFTS(ids, geoIDs, TSs,Xs,years):
    fig, AXS = plt.subplots(figsize = (20,16), ncols =2, nrows =2)
    for ID, axs in zip(ids,AXS):
        ts = TSs[np.where(np.array(geoIDs) == ID)[0]][0]; ts = np.log(ts)-np.log(TSs.mean(0))
        incr = Xs[np.where(np.array(geoIDs) == ID)[0]][0]
        axs[0].set_title('{}'.format(iso_dict[ID]), fontsize= 21, y=.5, x=-.35)
        for ax, vals, title in zip(axs, [ts,incr], ['centered log-mortality rates','smooth increments']):
            year_cmp = plt.cm.viridis(np.linspace(0,1,len(vals)))
            for _val , _color , _year in zip(vals,year_cmp, years[:-2] ):
                ax.plot(_val , label = _year, c=_color,alpha=.8);
            ax.set_ylabel(title,fontsize = 17);ax.set_xlabel('age',fontsize = 17);ax.set_xticks(np.linspace(0,_val.size,21))
            ax.set_xticklabels(np.linspace(0,100,21).astype(int), rotation = 45, fontsize = 15); ax.tick_params(axis='y', labelsize=15)
    ax.legend(loc='right', bbox_to_anchor=(1.3, 1.1), ncol=1,fontsize = 14); plt.subplots_adjust(hspace=0.4);

    return fig
