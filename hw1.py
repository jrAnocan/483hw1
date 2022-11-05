import cv2
import numpy as np
from cmath import log

def divergenceD(q,s,k_means):
    res = 0
    for i in range(k_means): 
        res += q[i]*log((q[i] / s[i]),10)
        
    return res

def divergenceJSD(q,s,k_means):
    res = (1/2) * divergenceD(q,np.divide( np.add(q,s), 2 ),k_means) + (1/2) * divergenceD(s, np.divide( np.add(q,s), 2 ), k_means)
    return res

def perChannel(source_cache, lines, queryNo, grid_dimension, k_means):
    length = len(lines)
    correct_comparision=0

    for x in range(length):
        #-------------------------------------------------------- FOR EACH IMAGE
        query_image_path = "dataset/query_"+queryNo+"/"+lines[x]
        query_image = np.array(cv2.imread(query_image_path))

        query_image = np.split(query_image,grid_dimension)
        for i in range(len(query_image)):
            query_image[i] = np.hsplit(query_image[i],grid_dimension)

        

        
        
        query_cache = [ [ [] for b in range(grid_dimension) ] for a in range(grid_dimension)]
        
        for i in range(grid_dimension):
            for j in range(grid_dimension):
                #-------------------------------------------------- FOR EACH GRID

                query_blue_column = np.zeros(k_means)
                query_green_column = np.zeros(k_means)
                query_red_column = np.zeros(k_means)

                for k in range(int(96/grid_dimension)):
                    for l in range(int(96/grid_dimension)):
                        query_blue_column[int(query_image[i][j][k][l][0] / (256/k_means))]+=1
                        query_green_column[int(query_image[i][j][k][l][1] / (256/k_means))]+=1
                        query_red_column[int(query_image[i][j][k][l][2] / (256/k_means))]+=1

                for a in range(k_means):
                    query_blue_column[a] = query_blue_column[a] / 9216
                    if(query_blue_column[a] == 0):
                        query_blue_column[a]+=1/9216
                    query_green_column[a] = query_green_column[a] / 9216
                    if(query_green_column[a] == 0):
                        query_green_column[a]+=1/9216
                    query_red_column[a] = query_red_column[a] / 9216
                    if(query_red_column[a] == 0):
                        query_red_column[a]+=1/9216
                
                query_cache[i][j] = [query_blue_column, query_green_column, query_red_column]
        #--------------------------------------------------------- COMPARE THE IMAGE WITH SOURCE_CACHE       
        
        name_of_the_similar_image=""
        min_divergence = 999999
        for i in range(length):   
            divergence = 0
            for j in range(grid_dimension):  
                for k in range(grid_dimension):
                    divergence += ( divergenceJSD(query_cache[j][k][0],source_cache[i][j][k][0],k_means) + divergenceJSD(query_cache[j][k][1],source_cache[i][j][k][1],k_means) + divergenceJSD(query_cache[j][k][2],source_cache[i][j][k][2],k_means) ) / (3*(grid_dimension**2))
            if(divergence < min_divergence):
                min_divergence = divergence
                name_of_the_similar_image = lines[i]

        if(name_of_the_similar_image == lines[x]):
            correct_comparision += 1    

    print(correct_comparision)      
               
                    
                    

if __name__=="__main__":
    queryNo = input("Enter query number(1,2,3) which is to be compared with support queries: ") 
    grid_dimension = int( input("Enter grid dimension (square grid): ") )
    k_means = int( input("Enter k_means (must divide 256): ") )

    with open("dataset/InstanceNames.txt") as file:
        lines = [line.rstrip() for line in file]

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
                source_blue_column = np.zeros(k_means)
                source_green_column = np.zeros(k_means)
                source_red_column = np.zeros(k_means)
                
                for k in range(int(96/grid_dimension)):
                    for l in range(int(96/grid_dimension)):
                        source_blue_column[int(source_image[i][j][k][l][0] / (256/k_means))]+=1
                        source_green_column[int(source_image[i][j][k][l][1] / (256/k_means))]+=1
                        source_red_column[int(source_image[i][j][k][l][2] / (256/k_means))]+=1

                for a in range(k_means):
                    source_blue_column[a] = source_blue_column[a] / 9216
                    if(source_blue_column[a] == 0):
                        source_blue_column[a]+=1/9216
                    source_green_column[a] = source_green_column[a] / 9216
                    if(source_green_column[a] == 0):
                        source_green_column[a]+=1/9216
                    source_red_column[a] = source_red_column[a] / 9216
                    if(source_red_column[a] == 0):
                        source_red_column[a]+=1/9216
                source_cache[x][i][j] = [source_blue_column, source_green_column, source_red_column]
            

    perChannel(source_cache, lines, queryNo, grid_dimension, k_means)
