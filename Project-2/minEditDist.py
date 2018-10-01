insertion_cost = 1
deletion_cost = 1
substitution_cost = 2
def minEditDist(source, target):
    n=len(source)
    m=len(target)

    #Create a distance matrix
    dist= [[0 for x in range(n+1)] for x in range(m+1)]
    
    #Initialization
    for i in range(0,n+1): 
        dist[i][0]=i
    for j in range(0,m+1):
        dist[0][j]=j
        
    # Recurrence relation:
    for i in range(1,n+1): 
        for j in range(1,m+1):

            # check if last characters are same
            if source[i-1] == target[j-1]: 
                dist[i][j] = dist[i-1][j-1]

            else: 
                dist[i][j] = min(dist[i][j-1]+ insertion_cost,   
                                   dist[i-1][j]+ deletion_cost,    
                                   dist[i-1][j-1]+ substitution_cost)  

    return dist[n][m]

source = "intention"
target = "execution"
  
print("Minimum edit distance between", source,  "and", target,  "is", minEditDist(source, target)) 
