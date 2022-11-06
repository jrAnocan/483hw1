import cv2
import numpy as np
from cmath import log

def divergenceD(q,s,k_means):
    res = 0
    norm_q=0
    norm_s=0
    for i in range(k_means):
        if(q[i]==0):
            q[i]+=1
        if(s[i]==0):
            s[i]+=1
        norm_q+=q[i]
        norm_s+=s[i]

    for i in range(k_means): 
        
        res += (q[i]/norm_q)*log(((q[i]/norm_q) / (s[i]/norm_s)),10)
        
    return res

def divergenceJSD(q,s,k_means):
    res = (1/2) * divergenceD(q,np.divide( np.add(q,s), 2 ),k_means) + (1/2) * divergenceD(s, np.divide( np.add(q,s), 2 ), k_means)
    return res

def divergence3DD(q,s,k_means,norm):
    res = 0
    for i in range(k_means): 
        for j in range(k_means):
            for k in range(k_means):
                if(q[i][j][k]>0 and s[i][j][k]>0):
                   
                    res += ((q[i][j][k])/norm) * log((((q[i][j][k])/norm) / ((s[i][j][k])/norm)),10)
                    
        
    return res

def divergence3DJSD(q, s,k_means,norm):
  
    res = (1/2) * divergence3DD(q,np.divide( np.add(q,s), 2 ),k_means, norm) + (1/2) * divergence3DD(s, np.divide( np.add(q,s), 2 ), k_means, norm)
    return res

def histogramPerChannel(source_cache, lines, queryNo, grid_dimension, k_means, type):
    length = len(lines)
    correct_comparision=0

    for x in range(length):
        #-------------------------------------------------------- FOR EACH IMAGE
        query_image_path = "dataset/query_"+queryNo+"/"+lines[x]
        query_image = np.array(cv2.imread(query_image_path))

        query_image = np.split(query_image,grid_dimension)
        for i in range(len(query_image)):
            query_image[i] = np.hsplit(query_image[i],grid_dimension)

        

        
        
        name_of_the_similar_image=""
        divergence = np.zeros(length,dtype="complex_")
        for i in range(grid_dimension):
            for j in range(grid_dimension):
                #-------------------------------------------------- FOR EACH GRID
                if(type==1):
                    query_blue_column = np.zeros(k_means)
                    query_green_column = np.zeros(k_means)
                    query_red_column = np.zeros(k_means)

                    for k in range(int(96/grid_dimension)):
                        for l in range(int(96/grid_dimension)):
                            query_blue_column[int(query_image[i][j][k][l][0] / (256/k_means))]+=1
                            query_green_column[int(query_image[i][j][k][l][1] / (256/k_means))]+=1
                            query_red_column[int(query_image[i][j][k][l][2] / (256/k_means))]+=1
            
                    
                    for a in range(length):
                        divergence[a] += ( divergenceJSD(query_blue_column, source_cache[a][i][j][0],k_means) + divergenceJSD(query_green_column, source_cache[a][i][j][1],k_means) + divergenceJSD(query_red_column, source_cache[a][i][j][2],k_means) ) / 3
                elif(type==2):
                    query_3D_histogram = np.zeros((k_means,k_means,k_means))
                    for k in range(int(96/grid_dimension)):
                        for l in range(int(96/grid_dimension)):
                            query_3D_histogram[int(query_image[i][j][k][l][0] / (256/k_means))][int(query_image[i][j][k][l][1] / (256/k_means))][int(query_image[i][j][k][l][2] / (256/k_means))] += 1
                   
                    for a in range(length):
                        divergence[a] += ( divergence3DJSD(query_3D_histogram, source_cache[a][i][j],k_means,(96/grid_dimension)**2) ) 
               
        #--------------------------------------------------------- COMPARE THE IMAGE WITH SOURCE_CACHE       
        min_index = np.argmin(divergence)
        if(min_index == x):
            correct_comparision += 1
        

    print(correct_comparision)      
               
                    

def cacheSource(type,lines,grid_dimension,k_means):

    length = len(lines)


    source_cache = [ [ [ [] for c in range(grid_dimension) ] for b in range(grid_dimension) ] for a in range(length) ]
    for x in range(length):
        source_image_path = "dataset/support_96/"+lines[x]
        source_image = np.array(cv2.imread(source_image_path))
        #----------------------------------------------------------------------- MAKE GRID    
        source_image = np.split(source_image,grid_dimension)

        for i in range(len(source_image)):
            source_image[i] = np.hsplit(source_image[i],grid_dimension)
            
        for i in range(grid_dimension):
            for j in range(grid_dimension):
                    #----------------------------------------------------------------------- FOR EACH GRID   
                if(type == 1): 
                    source_blue_column = np.zeros(k_means)
                    source_green_column = np.zeros(k_means)
                    source_red_column = np.zeros(k_means)
                        
                    for k in range(int(96/grid_dimension)):
                        for l in range(int(96/grid_dimension)):
                            source_blue_column[int(source_image[i][j][k][l][0] / (256/k_means))]+=1
                            source_green_column[int(source_image[i][j][k][l][1] / (256/k_means))]+=1
                            source_red_column[int(source_image[i][j][k][l][2] / (256/k_means))]+=1
                    
                    source_cache[x][i][j] = [source_blue_column, source_green_column, source_red_column]
                
                elif(type==2):
                    source_3D_histogram = np.zeros((k_means,k_means,k_means))
                
                    for k in range(int(96/grid_dimension)):
                        for l in range(int(96/grid_dimension)):
                            source_3D_histogram[int(source_image[i][j][k][l][0] / (256/k_means))][int(source_image[i][j][k][l][1] / (256/k_means))][int(source_image[i][j][k][l][2] / (256/k_means))] += 1
                    
                    source_cache[x][i][j] = source_3D_histogram
    
    return source_cache

if __name__=="__main__":

    queryNo = input("Enter query number(1,2,3) which is to be compared with support queries: ") 
    grid_dimension = int( input("Enter grid dimension (square grid): ") )
    k_means = int( input("Enter k_means (must divide 256): ") )
    type = int(input("Enter histogram type(1 for per channel, 2 for 3D): ") )

    with open("dataset/InstanceNames.txt") as file:
        lines = [line.rstrip() for line in file]

    source_cache = cacheSource(type,lines,grid_dimension,k_means)
    
    
    histogramPerChannel(source_cache, lines, queryNo, grid_dimension, k_means, type)
    
