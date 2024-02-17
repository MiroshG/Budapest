import networkx as nx
import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import scipy as sp
import itertools
from scipy.stats import zipf
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
#from scipy.stats import powerlaw
import powerlaw
from scipy.stats import binom
from sklearn.metrics import r2_score
import collections




path_to_zipped_file = 'network_12.zip' 

# open zip with network files
with zipfile.ZipFile(path_to_zipped_file) as z:
    # open file for the edges 
    with z.open('edges.csv') as f:
        df = pd.read_csv(f)
        df = df.rename(columns={"# source": " source"})
        for col in df.columns:
            df = df.rename(columns={col: col[1:]})
        # create network from edge list (call the proper network constructor, e.g., DiGrpah if directed)
        G = nx.from_pandas_edgelist(df)


def normal_cdf(x, mu, sigma):
    return norm.cdf(x, mu, sigma)
def normal_pdf(x,a,b):
    return norm.pdf(x,a,b)

def expon_cdf(x, a, b):
    return expon.cdf(x, a, b)
def expon_pdf(x,a,b):
    return expon.pdf(x,a,b)

def gamma_cdf(x, a, b):
    return gamma.cdf(x, a, b)
def gamma_pdf(x,a,b):
    return gamma.pdf(x,a,b)

def poisson_cdf(x, a):
    return poisson.cdf(x, a)
def poisson_pdf(x,a):
    return poisson.pmf(x,a)

def binom_cdf(x, a, b):
    return binom.cdf(x, a, b)
def binom_pdf(x,a,b):
    return binom.pdf(x,a,b)


def part_1(G):
    list_of_nodes=list(G.nodes())
    print('Number of nodes:')
    print(len(list_of_nodes))
    print(' ')

    list_of_edges=list(G.edges())
    print('Number of edges:')
    print(len(list_of_edges))
    print(' ')

    density_of_graph=nx.density(G)
    print('Density of the graph:')
    print(density_of_graph)
    print(' ')

    average_clustering=nx.average_clustering(G, weight=None, count_zeros=True)
    print('Average clustering coefficient:')
    print(average_clustering)
    print(' ')

    global_clustering=nx.transitivity(G)
    print('Global clustering:')
    print(global_clustering)
    print(' ')

    average_path_length=nx.average_shortest_path_length(G, weight=None, method=None)
    print('Average shortest path length:')
    print(average_path_length)
    print(' ')

    diameter_network=nx.diameter(G, e=None, usebounds=False, weight=None)
    print('Diameter of the network:')
    print(diameter_network)
    print(' ')

    degree_of_asortativity=nx.degree_pearson_correlation_coefficient(G)
    print('Degree of assortativity:')
    print(degree_of_asortativity)
    print(' ')

    bipartivity=nx.bipartite.spectral_bipartivity(G, nodes=None, weight='weight')
    print('Bipartivity index:')
    print(bipartivity)
    print(' ')

#part_1(G)
    
def degree_distribution(G):


    degree_sequence = sorted((d for n, d in G.degree()), reverse=False)
    unique_degrees, counts =np.unique(degree_sequence, return_counts=True)

    counts_array=np.array(counts)
    cumulative_sum=np.cumsum(counts_array)
    normalized_cumulative_sum=cumulative_sum/max(cumulative_sum)

    ########################################### PDF ####################################################

    parameters_normal_pdf, _= curve_fit(normal_pdf, unique_degrees, counts/np.sum(counts), p0=[1, 1]) 

    mu_est_norm_pdf, sigma_est_norm_pdf = parameters_normal_pdf

    fit_y_normal_pdf = normal_pdf(unique_degrees, mu_est_norm_pdf, sigma_est_norm_pdf)
    r2_normal_pdf = r2_score(counts/np.sum(counts), fit_y_normal_pdf)



    r2_normal_pdf=round(r2_normal_pdf,3)

    plt.figure()
    plt.scatter(unique_degrees, counts/np.sum(counts), s=5)
    #plt.plot(unique_degrees, fit_y_normal_pdf,label='Normal PDF, r2= '+str(r2_normal_pdf), color='red')
    plt.legend()
    plt.title('Real network')
    plt.xlabel("Degree, k")
    plt.ylabel("PDF, p(k)")
    plt.savefig('Degree distribution PDF .png', dpi=300, bbox_inches='tight')


    ############################################ CDF ###################################################

    parameters_normal, _= curve_fit(normal_cdf, unique_degrees, normalized_cumulative_sum, p0=[1, 1]) 
    parameters_exp, _= curve_fit(expon_cdf, unique_degrees, normalized_cumulative_sum, p0=[1, 1]) 
    parameters_gamma, _= curve_fit(gamma_cdf, unique_degrees, normalized_cumulative_sum, p0=[1, 1]) 

  
    mu_est_norm, sigma_est_norm = parameters_normal
    mu_est_exp, sigma_est_exp = parameters_exp
    mu_est_gamma, sigma_est_gamma = parameters_gamma
    
    fit_y_normal = normal_cdf(unique_degrees, mu_est_norm, sigma_est_norm)
    r2_normal = r2_score(normalized_cumulative_sum, fit_y_normal)

    fit_y_exp = expon_cdf(unique_degrees, mu_est_exp, sigma_est_exp)
    r2_exp = r2_score(normalized_cumulative_sum, fit_y_exp)

    fit_y_gamma = gamma_cdf(unique_degrees, mu_est_gamma, sigma_est_gamma)
    r2_gamma = r2_score(normalized_cumulative_sum, fit_y_gamma)

    r2_normal=round(r2_normal,3)
    r2_exp=round(r2_exp,3)
    r2_gamma=round(r2_gamma,3)
    
    plt.figure()
    plt.scatter(unique_degrees, normalized_cumulative_sum, s=5, label='Results')
    plt.plot(unique_degrees, fit_y_normal,label='Normal CDF, r2= '+str(r2_normal), color='red')
    plt.plot(unique_degrees, fit_y_exp,label='Exponential CDF, r2= '+str(r2_exp), color='green')
    plt.plot(unique_degrees, fit_y_gamma,label='Gamma CDF, r2= '+str(r2_gamma), color='black', linestyle='dotted')
    plt.legend()
    plt.title('Real network')
    plt.xlabel("Degree, k")
    plt.ylabel("CDF")
    plt.savefig('Degree distribution CDF .png', dpi=300, bbox_inches='tight')


#degree_distribution(G)
    




def top_25(G):

    ##################################### DEGREE ####################################################
    degree_dict=nx.degree_centrality(G)

    sorted_degree_dict = {k: v for k, v in sorted(degree_dict.items(), key=lambda item: item[1])}

    sorted_degree_list=list(sorted_degree_dict.keys()) 
    sorted_degree_values=list(sorted_degree_dict.values()) 


    print('Top 25 nodes by degree centrality:')
    print(sorted_degree_list[:25])
    print(' ')

    sorted_degree_dict_2={}
    for i in range(len(sorted_degree_list)):
        if i<25:
            sorted_degree_dict_2[sorted_degree_list[i]]=i
        else:
            sorted_degree_dict_2[sorted_degree_list[i]]=25

    nx.set_node_attributes(G, sorted_degree_dict_2, 'sorted_degree')
    

    ###################################### CLOSENESS ##################################################

    closeness_centrality_dict=nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)

    sorted_closeness_dict = {k: v for k, v in sorted(closeness_centrality_dict.items(), key=lambda item: item[1])}
    sorted_closeness_list=list(sorted_closeness_dict.keys())
    print('Top 25 nodes by closeness centrality:')
    print(sorted_closeness_list[:25])
    print(' ')

    sorted_closeness_dict_2={}
    for i in range(len(sorted_closeness_list)):
        if i<25:
            sorted_closeness_dict_2[sorted_closeness_list[i]]=i
        else:
            sorted_closeness_dict_2[sorted_closeness_list[i]]=25

    nx.set_node_attributes(G, sorted_closeness_dict_2, 'sorted_clos')

    ###################################### BETWEENNESS ##################################################

    betweenness_dict=nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
    sorted_betweenness_dict = {k: v for k, v in sorted(betweenness_dict.items(), key=lambda item: item[1])}
    betweenness_nodes_list=list(sorted_betweenness_dict.keys()) 
    print('Top 25 nodes by betweenness centrality:')
    print(betweenness_nodes_list[:25])
    print(' ')

    sorted_betweenness_dict_2={}
    for i in range(len(betweenness_nodes_list)):
        if i<25:
            sorted_betweenness_dict_2[betweenness_nodes_list[i]]=i
        else:
            sorted_betweenness_dict_2[betweenness_nodes_list[i]]=25

    nx.set_node_attributes(G, sorted_betweenness_dict_2, 'sorted_betweenness')

    ###################################### EIGENVECTOR ##################################################

    eigenvector_dict=nx.eigenvector_centrality(G, max_iter=10000, tol=1e-06, nstart=None, weight=None)
    sorted_eigenvector_dict = {k: v for k, v in sorted(eigenvector_dict.items(), key=lambda item: item[1])}
    eigenvector_nodes_list=list(sorted_eigenvector_dict.keys()) 
    print('Top 25 nodes by eigenvector centrality:')
    print(eigenvector_nodes_list[:25])
    print(' ')

    sorted_eigenvector_dict_2={}
    for i in range(len(eigenvector_nodes_list)):
        if i<25:
            sorted_eigenvector_dict_2[eigenvector_nodes_list[i]]=i
        else:
            sorted_eigenvector_dict_2[eigenvector_nodes_list[i]]=25

    nx.set_node_attributes(G, sorted_eigenvector_dict_2, 'sorted_eigenvector')

    ###################################### KATZ INDEX ##################################################
    katz_dict=nx.katz_centrality(G, alpha=0.001, beta=1.0, max_iter=10000, tol=1e-06, nstart=None, normalized=True, weight=None)
    sorted_katz_dict = {k: v for k, v in sorted(katz_dict.items(), key=lambda item: item[1])}
    katz_nodes_list=list(sorted_katz_dict.keys()) 
    print('Top 25 nodes by katz index centrality:')
    print(katz_nodes_list[:25])
    print(' ')

    sorted_katz_dict_2={}
    for i in range(len(katz_nodes_list)):
        if i<25:
            sorted_katz_dict_2[katz_nodes_list[i]]=i
        else:
            sorted_katz_dict_2[katz_nodes_list[i]]=25

    nx.set_node_attributes(G, sorted_katz_dict_2, 'sorted_katz')

    ###################################### PAGERANK ##################################################
    pagerank_dict=nx.pagerank(G, personalization=None, max_iter=10000, tol=1e-06, nstart=None, dangling=None)
    sorted_pagerank_dict = {k: v for k, v in sorted(pagerank_dict.items(), key=lambda item: item[1])}
    pagerank_nodes_list=list(sorted_pagerank_dict.keys()) 
    print('Top 25 nodes by pagerank centrality:')
    print(pagerank_nodes_list[:25])
    print(' ')

    sorted_pagerank_dict_2={}
    for i in range(len(pagerank_nodes_list)):
        if i<25:
            sorted_pagerank_dict_2[pagerank_nodes_list[i]]=i
        else:
            sorted_pagerank_dict_2[pagerank_nodes_list[i]]=25

    nx.set_node_attributes(G, sorted_pagerank_dict_2, 'sorted_pagerank')

    ###################################### SUBGRAPH CENTRALITY ##################################################

    subgraph_dict=nx.subgraph_centrality(G)
    sorted_subgraph_dict = {k: v for k, v in sorted(subgraph_dict.items(), key=lambda item: item[1])}
    subgraph_nodes_list=list(sorted_subgraph_dict.keys()) 
    print('Top 25 nodes by subgraph centrality:')
    print(subgraph_nodes_list[:25])
    print(' ')

    sorted_subgraph_dict_2={}
    for i in range(len(subgraph_nodes_list)):
        if i<25:
            sorted_subgraph_dict_2[subgraph_nodes_list[i]]=i
        else:
            sorted_subgraph_dict_2[subgraph_nodes_list[i]]=25

    nx.set_node_attributes(G, sorted_subgraph_dict_2, 'sorted_subgraph')

    nx.write_gml(G,"top_25.gml")

    return(sorted_degree_dict_2,sorted_betweenness_dict_2,sorted_closeness_dict_2,sorted_eigenvector_dict_2,sorted_subgraph_dict_2,sorted_pagerank_dict_2,sorted_katz_dict_2)

#d,b,c,e,s,p,k=top_25(G)
    

    
def erdos_renyi_and_barabasi(nodes, graph):

    degree_list=nx.degree(G)

    degree_array=[degree for _, degree in degree_list]


    # Actual graph ################################


    average_clustering=nx.average_clustering(graph, weight=None, count_zeros=True)
    global_clustering=nx.transitivity(graph)
    average_path_length=nx.average_shortest_path_length(graph, weight=None, method=None)
    diameter_network=nx.diameter(graph, e=None, usebounds=False, weight=None)
    degree_of_asortativity=nx.degree_pearson_correlation_coefficient(graph)
    bipartivity=nx.bipartite.spectral_bipartivity(graph, nodes=None, weight='weight')

    G_array=[np.mean(average_clustering),np.mean(global_clustering),np.mean(average_path_length),np.mean(diameter_network),np.mean(degree_of_asortativity),
                np.mean(bipartivity)]
    
    degree_sequence_real = sorted((d for n, d in G.degree()), reverse=False)
    _, counts_real =np.unique(degree_sequence_real, return_counts=True)

    # Erdos-Renyi ##################################

    p=G_array[0]

    average_clustering_list=[]
    global_clustering_list=[]
    average_path_length_list=[]
    diameter_list=[]
    
    asortativity_list=[]
    bipartivity_list=[]

    ER_PDF_dict={}

    mean_degree_ER=[]

    for i in range(10):

        G_ER=nx.erdos_renyi_graph(nodes, p, seed=None, directed=False)

        mean_degree_ER.append(np.mean(nx.degree(G_ER)))

        degree_sequence_ER = sorted((d for n, d in G_ER.degree()), reverse=False)
        unique_degrees_ER, counts_ER =np.unique(degree_sequence_ER, return_counts=True)


        for j in range(len(unique_degrees_ER)):
            if not unique_degrees_ER[j] in ER_PDF_dict:
                ER_PDF_dict[unique_degrees_ER[j]]=counts_ER[j]
            else:
                ER_PDF_dict[unique_degrees_ER[j]]+=counts_ER[j]


        average_clustering_ER=nx.average_clustering(G_ER, weight=None, count_zeros=True)
        average_clustering_list.append(average_clustering_ER)

        global_clustering_ER=nx.transitivity(G_ER)
        global_clustering_list.append(global_clustering_ER)

        average_path_length_ER=nx.average_shortest_path_length(G_ER, weight=None, method=None)
        average_path_length_list.append(average_path_length_ER)

        diameter_network_ER=nx.diameter(G_ER, e=None, usebounds=False, weight=None)
        diameter_list.append(diameter_network_ER)

        degree_of_asortativity_ER=nx.degree_pearson_correlation_coefficient(G_ER)
        asortativity_list.append(degree_of_asortativity_ER)

        bipartivity_ER=nx.bipartite.spectral_bipartivity(G_ER, nodes=None, weight='weight')
        bipartivity_list.append(bipartivity_ER)

    ER_array=[np.mean(average_clustering_list),np.mean(global_clustering_list),np.mean(average_path_length_list),np.mean(diameter_list),
              np.mean(asortativity_list),np.mean(bipartivity_list)]
    
    ER_std_array=[np.std(average_clustering_list),np.std(global_clustering_list), np.std(average_path_length_list),np.std(diameter_list),
              np.std(asortativity_list),np.std(bipartivity_list)]

    ER_rel_error=[]    

    for i in range(len(G_array)):
        if G_array[i]!=0:
            ER_rel_error.append(abs(G_array[i]-ER_array[i])*100/G_array[i])
        else:
            ER_rel_error.append('-')
    

    ER_PDF_dict=dict(sorted(ER_PDF_dict.items()))
    list_of_counts_PDF_ER=list(ER_PDF_dict.values())
    list_of_keys_PDF_ER=list(ER_PDF_dict.keys())

    

    counts_array_ER=np.array(list_of_counts_PDF_ER)
    cumulative_sum_ER=np.cumsum(counts_array_ER)
    normalized_cumulative_sum_ER=cumulative_sum_ER/max(cumulative_sum_ER)

    

    ########################################### PDF ####################################################

    plt.figure()
    plt.scatter(list_of_keys_PDF_ER, list_of_counts_PDF_ER/np.sum(list_of_counts_PDF_ER), s=5)
    plt.title('Erdös-Rényi')
    plt.xlabel("Degree, k")
    plt.ylabel("PDF, p(k)")
    plt.savefig('Degree distribution PDF ER.png', dpi=300, bbox_inches='tight')

    ################################################ CDF ##################################################

    """parameters_ER, _= curve_fit(poisson_cdf, list_of_keys_PDF_ER, normalized_cumulative_sum_ER, p0=[(nodes-1)*p]) 

    mu_est_ER= parameters_ER

    fit_y_ER = poisson_cdf(list_of_keys_PDF_ER, mu_est_ER)
    r2_ER = r2_score(normalized_cumulative_sum_ER, fit_y_ER) 
    r2_ER=round(r2_ER,3)"""


    plt.figure()
    plt.scatter(list_of_keys_PDF_ER, normalized_cumulative_sum_ER, s=5, label='Results')
    #plt.plot(list_of_keys_PDF_ER, fit_y_ER,label='Poisson CDF, r2= '+str(r2_ER), color='red')
    plt.legend()
    plt.title('Erdös-Rényi')
    plt.xlabel("Degree, k")
    plt.ylabel("CDF")
    plt.savefig('Degree distribution CDF ER.png', dpi=300, bbox_inches='tight')


    # Barabasi-Albert ######################################

    average_clustering_list=[]
    global_clustering_list=[]
    average_path_length_list=[]
    diameter_list=[]
    
    asortativity_list=[]
    bipartivity_list=[]

    BA_PDF_dict={}

    edges=np.mean(degree_array)

    edges=int(edges)

    mean_degree_BA=[]

    for i in range(10):

        G_BA=nx.barabasi_albert_graph(nodes, edges, seed=None, initial_graph=None)

        mean_degree_BA.append(np.mean(nx.degree(G_BA)))

        degree_sequence_BA = sorted((d for n, d in G_BA.degree()), reverse=False)
        unique_degrees_BA, counts_BA =np.unique(degree_sequence_BA, return_counts=True)

        for j in range(len(unique_degrees_BA)):
            if not unique_degrees_BA[j] in BA_PDF_dict:
                BA_PDF_dict[unique_degrees_BA[j]]=counts_BA[j]
            else:
                BA_PDF_dict[unique_degrees_BA[j]]+=counts_BA[j]


        average_clustering_BA=nx.average_clustering(G_BA, weight=None, count_zeros=True)
        average_clustering_list.append(average_clustering_BA)

        global_clustering_BA=nx.transitivity(G_BA)
        global_clustering_list.append(global_clustering_BA)

        average_path_length_BA=nx.average_shortest_path_length(G_BA, weight=None, method=None)
        average_path_length_list.append(average_path_length_BA)

        diameter_network_BA=nx.diameter(G_BA, e=None, usebounds=False, weight=None)
        diameter_list.append(diameter_network_BA)

        degree_of_asortativity_BA=nx.degree_pearson_correlation_coefficient(G_BA)
        asortativity_list.append(degree_of_asortativity_BA)
        bipartivity_BA=nx.bipartite.spectral_bipartivity(G_BA, nodes=None, weight='weight')
        bipartivity_list.append(bipartivity_BA)

    BA_array=[np.mean(average_clustering_list),np.mean(global_clustering_list),np.mean(average_path_length_list),np.mean(diameter_list),
              np.mean(asortativity_list),np.mean(bipartivity_list)]
    
    BA_std_array=[np.std(average_clustering_list),np.std(global_clustering_list),np.std(average_path_length_list),np.std(diameter_list),
              np.std(asortativity_list),np.std(bipartivity_list)]
    
    BA_rel_error=[]  

    for i in range(len(G_array)):
        if G_array[i]!=0:
            BA_rel_error.append(abs(G_array[i]-BA_array[i])*100/G_array[i])
        else:
            BA_rel_error.append('-')


    BA_PDF_dict=dict(sorted(BA_PDF_dict.items()))
    list_of_counts_PDF_BA=list(BA_PDF_dict.values())
    list_of_keys_PDF_BA=list(BA_PDF_dict.keys())

    counts_array_BA=np.array(list_of_counts_PDF_BA)
    cumulative_sum_BA=np.cumsum(counts_array_BA)
    normalized_cumulative_sum_BA=cumulative_sum_BA/max(cumulative_sum_BA)

    ########################################### PDF ####################################################

    plt.figure()
    plt.scatter(list_of_keys_PDF_BA, list_of_counts_PDF_BA/np.sum(list_of_counts_PDF_BA), s=5)
    plt.title('Barabási-Albert')
    plt.xlabel("Degree, k")
    plt.ylabel("PDF, p(k)")
    plt.savefig('Degree distribution PDF BA.png', dpi=300, bbox_inches='tight')

    ################################################ CDF ##################################################

    """parameters_BA, _= curve_fit(power_cdf, list_of_keys_PDF_BA, normalized_cumulative_sum_BA, p0=[-2], maxfev=5000) 

    value= parameters_BA

    fit_y_BA = 1-power_cdf(list_of_keys_PDF_BA, value)


    r2_BA = r2_score(normalized_cumulative_sum_BA, fit_y_BA) 

    r2_BA=round(r2_BA,3)"""

    plt.figure()
    plt.scatter(list_of_keys_PDF_BA, normalized_cumulative_sum_BA, s=5, label='Results')
    #plt.plot(list_of_keys_PDF_BA, fit_y_BA,label='Power law CDF, r2= '+str(r2_BA), color='red')
    plt.legend()
    plt.title('Barabási-Albert')
    plt.xlabel("Degree, k")
    plt.ylabel("CDF")
    plt.savefig('Degree distribution CDF BA.png', dpi=300, bbox_inches='tight')

    ################################## WATTS STROGATZ ####################################################################

    average_clustering_list=[]
    global_clustering_list=[]
    average_path_length_list=[]
    diameter_list=[]
    
    asortativity_list=[]
    bipartivity_list=[]

    WS_PDF_dict={}

    mean_degree=int(np.mean(degree_array))


    mean_degree_WS=[]

    for i in range(10):

        G_WS=nx.watts_strogatz_graph(nodes, mean_degree, 1, seed=None)

        mean_degree_WS.append(np.mean(nx.degree(G_WS)))

        degree_sequence_WS = sorted((d for n, d in G_WS.degree()), reverse=False)
        unique_degrees_WS, counts_WS =np.unique(degree_sequence_WS, return_counts=True)

        for j in range(len(unique_degrees_WS)):
            if not unique_degrees_WS[j] in WS_PDF_dict:
                WS_PDF_dict[unique_degrees_WS[j]]=counts_WS[j]
            else:
                WS_PDF_dict[unique_degrees_WS[j]]+=counts_WS[j]


        average_clustering_WS=nx.average_clustering(G_WS, weight=None, count_zeros=True)
        average_clustering_list.append(average_clustering_WS)

        global_clustering_WS=nx.transitivity(G_WS)
        global_clustering_list.append(global_clustering_WS)

        average_path_length_WS=nx.average_shortest_path_length(G_WS, weight=None, method=None)
        average_path_length_list.append(average_path_length_WS)

        diameter_network_WS=nx.diameter(G_WS, e=None, usebounds=False, weight=None)
        diameter_list.append(diameter_network_WS)

        degree_of_asortativity_WS=nx.degree_pearson_correlation_coefficient(G_WS)
        asortativity_list.append(degree_of_asortativity_WS)
        bipartivity_WS=nx.bipartite.spectral_bipartivity(G_WS, nodes=None, weight='weight')
        bipartivity_list.append(bipartivity_WS)

    WS_array=[np.mean(average_clustering_list),np.mean(global_clustering_list),np.mean(average_path_length_list),np.mean(diameter_list),
              np.mean(asortativity_list),np.mean(bipartivity_list)]
    
    WS_std_array=[np.std(average_clustering_list),np.std(global_clustering_list),np.std(average_path_length_list),np.std(diameter_list),
              np.std(asortativity_list),np.std(bipartivity_list)]
    
    WS_rel_error=[]  

    for i in range(len(G_array)):
        if G_array[i]!=0:
            WS_rel_error.append(abs(G_array[i]-WS_array[i])*100/G_array[i])
        else:
            WS_rel_error.append('-')

    WS_PDF_dict=dict(sorted(WS_PDF_dict.items()))
    list_of_counts_PDF_WS=list(WS_PDF_dict.values())
    list_of_keys_PDF_WS=list(WS_PDF_dict.keys())

    counts_array_WS=np.array(list_of_counts_PDF_WS)
    cumulative_sum_WS=np.cumsum(counts_array_WS)
    normalized_cumulative_sum_WS=cumulative_sum_WS/max(cumulative_sum_WS)

    ########################################### PDF ####################################################

    plt.figure()
    plt.scatter(list_of_keys_PDF_WS, list_of_counts_PDF_WS/np.sum(list_of_counts_PDF_WS), s=5)
    plt.title('Watts-Strogatz')
    plt.xlabel("Degree, k")
    plt.ylabel("PDF, p(k)")
    plt.savefig('Degree distribution PDF WS.png', dpi=300, bbox_inches='tight')

    ################################################ CDF ##################################################

    """parameters_WS, _= curve_fit(poisson_cdf, list_of_keys_PDF_WS, normalized_cumulative_sum_WS, p0=[140]) 

    mu_est_WS= parameters_WS

    fit_y_WS = poisson_cdf(list_of_keys_PDF_WS, mu_est_WS)
    r2_WS = r2_score(normalized_cumulative_sum_WS, fit_y_WS) 
    r2_WS=round(r2_WS,3)"""

    plt.figure()
    plt.scatter(list_of_keys_PDF_WS, normalized_cumulative_sum_WS, s=5, label='Results')
    #plt.plot(list_of_keys_PDF_WS, fit_y_WS,label='Gamma CDF, r2= '+str(r2_WS), color='red')
    plt.legend()
    plt.title('Watts-Strogatz')
    plt.xlabel("Degree, k")
    plt.ylabel("CDF")
    plt.savefig('Degree distribution CDF WS.png', dpi=300, bbox_inches='tight')


    ####################################################

    text_array=['AVERAGE CLUSTERING','GLOBAL CLUSTERING','AVERAGE PATH LENGTH','DIAMETER','DEGREE OF ASORTATIVITY','BIPARTIVITY']

    data = np.array([text_array, G_array, ER_array, ER_std_array, BA_array, BA_std_array,WS_array, WS_std_array]).T

    np.savetxt('Random_networks.txt', data, fmt='%s', delimiter='\t', header='Values\tActual_network\tER\tstd_ER\tBA\tstd_BA\tWS\tstd_WS', comments='')

erdos_renyi_and_barabasi(1015,G)

def part_3(G):

    modularity_maximization=nx.community.greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1, best_n=None)
    

    modularity_disorder_dict={}

    first_cluster=modularity_maximization[0]
    second_cluster=modularity_maximization[1]

    print('modularity maximization')
    print('G: '+str(nx.community.modularity(G, modularity_maximization, weight='weight', resolution=1)))
    

    first_cluster=list(first_cluster)
    second_cluster=list(second_cluster)

    for i in first_cluster:
        modularity_disorder_dict[i]=1

    for i in second_cluster:
        modularity_disorder_dict[i]=2


    nx.set_node_attributes(G, modularity_disorder_dict, 'modularity_maximization')



    modularity_maximization_2=nx.community.greedy_modularity_communities(G, weight=None, resolution=1, cutoff=6, best_n=6)

    print('modularity maximization 6')
    print('G: '+str(nx.community.modularity(G, modularity_maximization_2, weight='weight', resolution=1)))




    louvain_method=nx.community.louvain_communities(G, weight='weight', resolution=1, threshold=1e-07, seed=None)

    print('Louvain method')
    print('G: '+str(nx.community.modularity(G, louvain_method, weight='weight', resolution=1)))

    first_cluster_louvain=louvain_method[0]
    second_cluster_louvain=louvain_method[1]
    third_cluster_louvain=louvain_method[2]
    fourth_cluster_louvain=louvain_method[3]

    louvain_dict={}

    for i in first_cluster_louvain:
        louvain_dict[i]=1

    for i in second_cluster_louvain:
        louvain_dict[i]=2

    for i in third_cluster_louvain:
        louvain_dict[i]=3

    for i in fourth_cluster_louvain:
        louvain_dict[i]=4

    nx.set_node_attributes(G, louvain_dict, 'louvain')

    kernighan_lin_bisection=nx.community.kernighan_lin_bisection(G, partition=None, max_iter=10, weight='weight', seed=None)

    print('Kernighan-lin method')
    print('G: '+str(nx.community.modularity(G, kernighan_lin_bisection, weight='weight', resolution=1)))

    kernighan_dict={}
    first_cluster_kernighan=kernighan_lin_bisection[0]
    second_cluster_kernighan=kernighan_lin_bisection[1]



    for i in first_cluster_kernighan:
        kernighan_dict[i]=1
    for i in second_cluster_kernighan:
        kernighan_dict[i]=2

    nx.set_node_attributes(G, kernighan_dict, 'kernighan_lin')


    """girvan_newman=nx.community.girvan_newman(G, most_valuable_edge=None)

    communities_2 = None
    communities_3 = None
    communities_4 = None

    for communities in girvan_newman:
        if len(communities) == 2 and communities_2 is None:
            communities_2 = communities
        elif len(communities) == 3 and communities_3 is None:
            communities_3 = communities
        elif len(communities) == 4 and communities_4 is None:
            communities_4 = communities

        if communities_2 is not None and communities_3 is not None and communities_4 is not None:
            break


    list_of_communities_2 = [list(community) for community in communities_2]
    list_of_communities_3 = [list(community) for community in communities_3]
    list_of_communities_4 = [list(community) for community in communities_4]

    print('2')
    print('G: '+str(nx.community.modularity(G, list_of_communities_2, weight='weight', resolution=1)))
    print('3')
    print('G: '+str(nx.community.modularity(G, list_of_communities_3, weight='weight', resolution=1)))
    print('4')
    print('G: '+str(nx.community.modularity(G, list_of_communities_4, weight='weight', resolution=1)))


    first_2=list_of_communities_2[0]
    second_2=list_of_communities_2[1]

    dict_2={}

    for i in first_2:
        dict_2[i]=1
    for i in second_2:
        dict_2[i]=2



    first_3=list_of_communities_3[0]
    second_3=list_of_communities_3[1]
    third_3=list_of_communities_3[2]

    nx.set_node_attributes(G, dict_2, 'girvan_newman_2')

    dict_3={}

    for i in first_3:
        dict_3[i]=1
    for i in second_3:
        dict_3[i]=2
    for i in third_3:
        dict_3[i]=3

    first_4=list_of_communities_4[0]
    second_4=list_of_communities_4[1]
    third_4=list_of_communities_4[2]
    fourth_4=list_of_communities_4[3]

    nx.set_node_attributes(G, dict_3, 'girvan_newman_3')

    dict_4={}

    for i in first_4:
        dict_4[i]=1
    for i in second_4:
        dict_4[i]=2
    for i in third_4:
        dict_4[i]=3
    for i in fourth_4:
        dict_4[i]=4

    nx.set_node_attributes(G, dict_4, 'girvan_newman_4')"""
    
    nx.write_gml(G,"network.gml")

    return louvain_dict



#list_dict=part_3(G)   



# open zip with network files
with zipfile.ZipFile(path_to_zipped_file) as z:
    # open file for the edges 
    with z.open('edges.csv') as f:
        df = pd.read_csv(f)
        df = df.rename(columns={"# source": " source"})
        for col in df.columns:
            df = df.rename(columns={col: col[1:]})
        # create network from edge list (call the proper network constructor, e.g., DiGrpah if directed)
        F = nx.from_pandas_edgelist(df)
        
    # open file for the graph properties (this one was a little tricky...)
    parsed_data = []
    with z.open('gprops.csv') as f:
        for line in f:
            decoded_line = line.decode('utf-8')
            # Skip empty lines or lines starting with '#'
            if not decoded_line.strip() or decoded_line.startswith('#'):
                continue
        
            # Split the line by the first comma only, since only 2 fields are expected
            split_line = decoded_line.split(',', 1)
        
            # Check if the line is split into exactly 2 parts
            if len(split_line) == 2:
                # Trim and strip quotes from the second part if necessary
                key, value = split_line
                value = value.strip().strip('"')
                parsed_data.append([key.strip(), value])

    # adding the data to the network object
    for prop,value in parsed_data:
        F.graph[prop] = value
        
    # open file for node properties and add them to the graph
    with z.open('nodes.csv') as f:
        df = pd.read_csv(f)
        df = df.rename(columns={"# index": " id"})
        for col in df.columns:
            df = df.rename(columns={col: col[1:]})
        for att in list(df.columns[1:]):
            att_dict = pd.Series(df[att].values,index=df.id).to_dict()
            nx.set_node_attributes(F,att_dict,att)


def print_name_and_region(dictionary,names,values, number):
    order=list(dictionary.keys())

    region_dict=dict(sorted(values.items(), key=lambda x: order.index(x[0])))
    name_dict=dict(sorted(names.items(), key=lambda x: order.index(x[0])))

    region_list=list(region_dict.values())
    name_list=list(name_dict.values())

    print(order[:number])
    print(region_list[:number])
    print(name_list[:number])
    print(' ')

def anatomic_place(degree_top,betweenness_top,closeness_top,eigenvector_top,subgraph_top,pagerank_top,katz_top, number):

    attribute_names = nx.get_node_attributes(F, 'dn_name')
    attribute_values = nx.get_node_attributes(F, 'dn_region')


    list_to_iterate=[degree_top,betweenness_top,closeness_top,eigenvector_top,subgraph_top,pagerank_top,katz_top]
    list_of_text=['degree','betweenness','closeness','eigenvector','subgraph','pagerank','katz']

    
    print('')

    for i in range(len(list_to_iterate)):
        print(list_of_text[i])
        print_name_and_region(list_to_iterate[i],attribute_names,attribute_values, number)

#anatomic_place(d,b,c,e,s,p,k, 5)
        
def connection_between_hemispheres():


    result_dict = {}

    counter_edges_between_hemispheres=0

    for edge in F.edges():
        node1, node2 = edge
        attr1 = F.nodes[node1]['dn_hemisphere']
        attr2 = F.nodes[node2]['dn_hemisphere']

        if attr1 == 'left' and attr2 == 'left':
            result_dict[node1] = 0
            result_dict[node2] = 0
        elif attr1 == 'left' or attr2 == 'right':
            result_dict[node1] = 1
            result_dict[node2] = 1
            counter_edges_between_hemispheres+=0.5
        elif attr1 == 'right' and attr2 == 'right':
            result_dict[node1] = 2
            result_dict[node2] = 2
        elif attr1 == 'right' or attr2 == 'left':
            result_dict[node1] = 3
            result_dict[node2] = 3
            counter_edges_between_hemispheres+=0.5

    print('number of edges between hemispheres: '+str(counter_edges_between_hemispheres))


    nx.set_node_attributes(G, result_dict, 'connections')
    nx.write_gml(G,"hemispheres.gml")

#connection_between_hemispheres()

def anatomic_modularity(mod):

    attribute_names = nx.get_node_attributes(F, 'dn_name')
    attribute_values = nx.get_node_attributes(F, 'dn_region')

    mod_list=['modularity', 'kernighan','louvain']
    counter=0
    mod_keys=list(mod.keys())
    mod_values=list(mod.values())

    unique_values=set(mod_values)

    region_dict=dict(sorted(attribute_values.items(), key=lambda x: mod_keys.index(x[0])))
    name_dict=dict(sorted(attribute_names.items(), key=lambda x: mod_keys.index(x[0])))

    region_list=list(region_dict.values())
    name_list=list(name_dict.values())

    different_regions=[]
    different_names=[]

    print(mod_list[counter])
    counter+=1

    for values in unique_values:
        positions = [index for index, value in enumerate(mod_values) if value == values]

        for i in positions:
            region=region_list[i]
            name=name_list[i]
            if region not in different_regions:
                different_regions.append(region)
            if name not in different_names:
                different_names.append(name)
        
        print('names')
        print(different_names)
        print('regions')
        print(different_regions)


#anatomic_modularity(list_dict)
