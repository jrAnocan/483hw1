from cmath import log
import numpy as np
import cv2

def chunkIt(seq, num):
    avg = 96 / float(num)
    out = []
    last = 0.0

    while last < 96:
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def divergenceD(q,s,k_means):
    res = 0
    for i in range(k_means):
        
        res += q[i]*log((q[i] / s[i]),10)
        
    return res

def divergenceJSD(q,s,k_means):
    res = (1/2) * divergenceD(q,np.divide( np.add(q,s), 2 ),k_means) + (1/2) * divergenceD(s, np.divide( np.add(q,s), 2 ), k_means)
    return res
    

def perChannel(k_means, queryNo,source_cache,lines):

    length = len(lines)

    correct_comparision=0
    for l1 in lines:

        min_divergence = 9999999999
        closest_picture_name=""
        query_image_path = "dataset/query_"+queryNo+"/"+l1
        query_image = np.array(cv2.imread(query_image_path))

        query_blue_column = np.zeros(k_means)
        query_green_column = np.zeros(k_means)
        query_red_column = np.zeros(k_means)

        for i in range(96):
            for j in range(96):
                query_blue_column[int(query_image[i][j][0] / (256/k_means))]+=1
                query_green_column[int(query_image[i][j][1] / (256/k_means))]+=1
                query_red_column[int(query_image[i][j][2] / (256/k_means))]+=1

        for i in range(k_means):
            query_blue_column[i] = query_blue_column[i] / 9216
            if(query_blue_column[i] == 0):
                query_blue_column[i]+=1/9216
            query_green_column[i] = query_green_column[i] / 9216
            if(query_green_column[i] == 0):
                query_green_column[i]+=1/9216
            query_red_column[i] = query_red_column[i] / 9216
            if(query_red_column[i] == 0):
                query_red_column[i]+=1/9216
                
        for i in range(length):
            divergence = (divergenceJSD(query_blue_column, source_cache[i][0],k_means) + divergenceJSD(query_green_column, source_cache[i][1],k_means) + divergenceJSD(query_red_column, source_cache[i][2],k_means))
            
            if(divergence < min_divergence):
                min_divergence = divergence
                closest_picture_name = lines[i]
        if(closest_picture_name == l1):
            correct_comparision += 1
    print(correct_comparision)
if __name__=="__main__":

    queryNo = input("Enter query number(1,2,3) which is to be compared with support queries: ") 
    k_means = int( input("Enter k_means (must divide 256): ") )

    with open("dataset/InstanceNames.txt") as file:
        lines = [line.rstrip() for line in file]

    
    length = len(lines)
    source_cache = [[] for i in range(length) ]

    for x in range(length):
        source_image_path = "dataset/support_96/"+lines[x]
        source_image = np.array(cv2.imread(source_image_path))

        source_blue_column = np.zeros(k_means)
        source_green_column = np.zeros(k_means)
        source_red_column = np.zeros(k_means)

        for i in range(96):
            for j in range(96):
                source_blue_column[int(source_image[i][j][0] / (256/k_means))]+=1
                source_green_column[int(source_image[i][j][1] / (256/k_means))]+=1
                source_red_column[int(source_image[i][j][2] / (256/k_means))]+=1

        for i in range(k_means):
            source_blue_column[i] = source_blue_column[i] / 9216
            if(source_blue_column[i] == 0):
                source_blue_column[i]+=1/9216
            source_green_column[i] = source_green_column[i] / 9216
            if(source_green_column[i] == 0):
                source_green_column[i]+=1/9216
            source_red_column[i] = source_red_column[i] / 9216
            if(source_red_column[i] == 0):
                source_red_column[i]+=1/9216
        source_cache[x] = [source_blue_column, source_green_column, source_red_column]
    
    """for i in range(length):
        for j in range(3):
            for k in range(k_means):
                if(source_cache[i][j][k] == 0):
                    print(i,j,k)
    """
        
    #print(source_cache[8][0][3])
    perChannel(k_means,queryNo, source_cache,lines)
    
