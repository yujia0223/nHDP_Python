    
def rev_cumsum(a = None): 
    d1,d2 = a.shape
    if d1 > d2:
        a = flipud(a)
        vec = cumsum(a)
        vec = flipud(vec)
    else:
        a = fliplr(a)
        vec = cumsum(a)
        vec = fliplr(vec)
    
    return vec