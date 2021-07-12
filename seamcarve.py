import cv2
import sys
import numpy as np

# Find a horizontal or vertical seam. seamDir is 1 for a horizontal seam, 0 otherwise.
def find_seam(M, seamDir):

    if seamDir == 1: # horizontal seam
        M = np.transpose(M) # We turn it into a vertical seam problem if this is the case
    path = []
    (h,w) = np.shape(M)

    min_idx = np.argmin(M[-1, :])
    i = h-1
    j = min_idx

    if seamDir == 1:
            path.append((j, i)) #need transpose for path if its actually horizontal
    else:
        path.append((i, j))

    while i != 0:
        prev_energies = []
        pos = []
        if j == 0:
            prev_energies.append(M[i-1,j])
            pos.append(j)
            prev_energies.append(M[i-1,j+1])
            pos.append(j + 1)
        elif j == w - 1:
            prev_energies.append(M[i-1,j])
            pos.append(j)
            prev_energies.append(M[i-1,j - 1])
            pos.append(j - 1)
        else:
            prev_energies.append(M[i-1,j])
            pos.append(j)
            prev_energies.append(M[i-1,j + 1])
            pos.append(j + 1)
            prev_energies.append(M[i-1,j - 1])
            pos.append(j - 1)

        minval = np.argmin(np.array([prev_energies]))
        j = pos[minval]
        i = i - 1
        if seamDir == 1:
            path.append((j, i)) #need transpose for path if its actually horizontal
        else:
            path.append((i, j))

    return path


# Remove a vertical seam from an image
def remove_vertical_seam(image, path):

    if np.ndim(image) == 3:
        (h,w,z) = np.shape(image)
        output = np.zeros((h, w - 1, 3))

        #Create new image with that seam removed
        #print(path)
        for (i, j) in path:
            #print((i,j))
            output[i, 0:j, :] = image[i, 0:j, :]
            output[i, j:, :] = image[i, j+1:, :]

        return output
    else:
        (h,w) = np.shape(image)
        output = np.zeros((h, w - 1))

        #Create new image with that seam removed
        #print(path)
        for (i, j) in path:
            #print((i,j))
            output[i, 0:j] = image[i, 0:j]
            output[i, j:] = image[i, j+1:]

        return output

# Remove a horizontal seam from an image
def remove_horizontal_seam(image, path):

    # print(path)
    if np.ndim(image) == 3:
        (h,w,z) = np.shape(image)
        output = np.zeros((h - 1, w, 3))

        #Create new image with that seam removed
        #print(path)
        #print(path)\
        #print(np.shape(output))
        #print(np.shape(image))
        for (i, j) in path:
            #print((i,j))
            output[0:i, j, :] = image[0:i, j, :]
            #print(i,j)
            #print(output[i:-1, j, :], image[i+1:-1, j, :])
            output[i:, j, :] = image[i+1:, j, :]

        return output
    else:
        (h,w) = np.shape(image)
        output = np.zeros((h - 1, w))

        #Create new image with that seam removed
        #print(path)
        for (i, j) in path:
            #print((i,j))
            output[0:i, j] = image[0:i, j]
            output[i:, j] = image[i+1:, j]

        return output

# This finds the energy map, the sum of the absolute value of gradient in x and y
def find_energy_map(image):

    out_x = np.abs(cv2.Sobel(image,cv2.CV_64F, 1, 0, ksize = 3, borderType = cv2.BORDER_REFLECT))
    out_y = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3, borderType = cv2.BORDER_REFLECT))
    out = out_x + out_y
    out = np.sum(out, axis = 2)
    return out

# This function finds M, the accumulated cost matrix. Tracing this matrix via the lowest values gives one the lowest energy seam.
def find_accumulated_cost(image): # Vertical

    energy_map = find_energy_map(image)

    M = np.zeros_like(energy_map)
    M[0,:] = energy_map[0,:]
    #print("First row of M")

    (h,w) = np.shape(energy_map)

    for i in range(1,h):
        for j in range(w):
            prev_energies = []
            if j == 0:
                prev_energies.append(M[i-1,j])
                prev_energies.append(M[i-1,j+1])
            elif j == w - 1:
                prev_energies.append(M[i-1,j])
                prev_energies.append(M[i-1,j - 1])
            else:
                prev_energies.append(M[i-1,j])
                prev_energies.append(M[i-1,j + 1])
                prev_energies.append(M[i-1,j - 1])
            M[i,j]  = energy_map[i,j] + min(prev_energies)
    #print(np.shape(M))
    return M


"""def find_transport_map(image, r, c):
    T = np.zeros((r,c))
    TBM = np.zeros((r,c))
    T[0,0] = 0

    M = find_accumulated_cost(image)

    E_h = np.min(M[:, -1])
    E_v = np.min(M[-1, :])

    # Initialize borders of T
    for j in range(c):
        T[0,j] = T[0, j-1] + E_v
        TBM[0,j] = 1

    for i in range(r):
        T[i,0] = T[i - 1, 0] + E_h
        TBM[i,0] = 0

    # Fill in T and TBM
    for i in range(1, r):
        imagewithoutrow = image
        for j in range(1, c):
            M = find_accumulated_cost(imagewithoutrow)

            E_h = np.min(M[:, -1])
            E_v = np.min(M[-1, :])

            tVertical = T[i - 1, j] + E_v
            tHorizontal = T[i, j - 1] + E_h

            if (tVertical < tHorizontal):
                T[i, j] = tVertical
                TBM[i, j] = 1
            else:
                T[i, j] = tHorizontal
                TBM[i, j] = 0
            path = find_seam(M, 0)
            imagewithoutrow = remove_vertical_seam(image, path)
        M = find_accumulated_cost(image)
        path = find_seam(M, 1)
        imagewithoutrow = remove_horizontal_seam(image, path)
        print(T)
    return T, TBM

"""

"""dsef find_transport_map(image, r, c): #Adapted from https://github.com/KirillLykov/cvision-algorithms/blob/master/src/seamCarving.m
    T = np.zeros((r + 1, c + 1), dtype=np.float64)
    TBM = np.ones_like(T) * -1

    # Fill in borders
    imagenorow = image

    for i in range(1, r):
        #Horizontal seams

        M = find_accumulated_cost(imagenorow)
        min_horizontal_seam = min(M[:,-1])
        #print(M[:,-1])
        seam = find_seam(M, 1)
        imagenorow = remove_horizontal_seam(imagenorow, seam)

        TBM[i, 0] = 0
        T[i, 0] = T[i - 1, 0] + min_horizontal_seam


    imagenocolumn = image

    for i in range(1, c):
        M = find_accumulated_cost(imagenocolumn)
        min_vertical_seam = min(M[-1,:])
        seam = find_seam(M, 0)
        imagenocolumn=remove_vertical_seam(imagenocolumn, seam)

        TBM[0, i] = 1
        T[0, i] = T[0, i - 1] + min_vertical_seam

    # Remove one row and one column

    M = find_accumulated_cost(imagenorow)
    min_horizontal_seam = min(M[:,-1])
    seam = find_seam(M, 1)
    imagenorow = remove_horizontal_seam(imagenorow, seam)

    M = find_accumulated_cost(imagenocolumn)
    min_vertical_seam = min(M[-1,:])
    seam = find_seam(M, 0)
    imagenocolumn=remove_vertical_seam(imagenocolumn, seam)

    print(T)
    # Fill in T
    for i in range(1,r):
        imagewithoutrow = image
        for j in range(1, c):
            M = find_accumulated_cost(imagewithoutrow)

            min_vertical_seam = min(M[-1,:])
            seam = find_seam(M, 1)
            imagenorow = remove_horizontal_seam(imagewithoutrow, seam)

            min_horizontal_seam = min(M[:,-1])
            seam = find_seam(M, 0)
            imagenocolumn = remove_vertical_seam(imagewithoutrow, seam)

            neighbors = [ T[i - 1, j] + min_horizontal_seam, T[i, j - 1] + min_vertical_seam]
            val = min(neighbors)
            idx = np.argmin(neighbors)

            T[i, j] = val
            TBM[i, j] = idx

            imagewithoutrow = imagenocolumn
        M = find_accumulated_cost(image)
        seam = find_seam(M, 1)
        image = remove_horizontal_seam(image, seam)
        print(T)

    return T, TBM

def find_order_of_operations(T, TBM):
    path = []
    (r,c) = np.shape(TBM)

    i = r
    j = c

    while i != 0 and j != 0:
        path.append(TBM[i,j])
        if TBM[i,j] == 0:
            i = i- 1
        else:
            j = j - 1

    return reverse(path)"""


def main():

    # Read in required input information
    args = sys.argv[1:]
    if len(args) < 3:
        print("Input: python seamcarve IMAGE NEW_H, NEW_W")

    image = cv2.imread(args[0])
    new_h = int(args[1])
    new_w = int(args[2])


    (h,w,z) = np.shape(image)
    if new_h > h or new_w > w:
        print("New h/w has to be smaller than the original h/w")
        quit(1)
    r = h - new_h
    c = w - new_w
    M = find_accumulated_cost(image)
    # Compute image energy map

    #This code is too slow, but finds the transport map and does optimal removal
    """T, TBM = find_transport_map(image, r, c)
    print(TBM)
    #Find order to remove seams
    order = find_order_of_operations(T, TBM)"""

    rows_to_go = r
    cols_to_go = c

    """output = image
    for val in order:
        if val == 1:
            path = find_seam(M, 0)
            output = remove_vertical_seam(image, path)
            M = find_accumulated_cost(output)
        elif val == 0:
            path = find_seam(M, 1)
            output = remove_horizontal_seam(image, path)
            M = find_accumulated_cost(output)"""
    #Remove seams in alternating order until one runs out.
    while rows_to_go != 0 or cols_to_go != 0:

        if rows_to_go != 0:
            M = find_accumulated_cost(np.transpose(image, [1, 0, 2]))
            M = np.transpose(M)
            min_horizontal_seam = min(M[:,-1])
            seam = find_seam(M, 1 )
            image = remove_horizontal_seam(image, seam)
            rows_to_go -= 1

        if cols_to_go != 0:
            M = find_accumulated_cost(image)
            min_horizontal_seam = min(M[-1,:])
            seam = find_seam(M, 0)
            image = remove_vertical_seam(image, seam)
            cols_to_go -= 1

        print(np.shape(image))
    image.astype(np.uint8)
    cv2.imwrite("output.jpg", image )
    return 0




if __name__ == "__main__":
    main()
