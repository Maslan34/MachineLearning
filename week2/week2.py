

######################################--------RECAP NUMPY---------#####################################

#importing
import numpy as np 


array = np.array([1,2,3]) #creating a numpy array
                          #1x3 vector

print(array)



array2= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

print(array2.shape) #getting vector size of numpy array
                    #(15,) -> 15x1 vector
                    
                    
a = array2.reshape(3,5) 

#vector was converted to a 3-row 5-column matrix.

print(a)




print("shape: ",a.shape) #(3, 5)

print("dimension: ",a.ndim) #getting dimension of the array.
                            #dimension:  2

print("data type: ",a.dtype.name) #data type:  int32
#getting dtype of each columns's data type


                       #!!!!!
print("size: ",a.size) #"size" function returns the dimension of the array before "reshape" function was applied.
                       #size:  15
                       
print("type",type(a))  #type <class 'numpy.ndarray'>




array3= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) #Size can be converted like that without executing reshape function.



print(array3) #array([[ 1,  2,  3,  4],
                     #[ 5,  6,  7,  8],
                     #[ 9, 10, 11, 12]])
                     
                    
print(array3.shape) #(3, 4)



zeros= np.zeros((3,4)) #3-row  5-columns matrix that filled with 0.

zeros[0,0]=5
print(zeros)



np.ones((3,4)) #3-row  4-columns matrix that filled with 1.




np.empty((4,5)) #creating an empty array.




np.arange(10,50,5) #Numbers between 10 and 50 are printed 5 by 5.

#array([10, 15, 20, 25, 30, 35, 40, 45])


a= np.linspace(0,10,20) #Numbers between 0 and 10 are printed at equal intervals.
#array([ 0.        ,  0.52631579,  1.05263158,  1.57894737,  2.10526316,
#       2.63157895,  3.15789474,  3.68421053,  4.21052632,  4.73684211,
#       5.26315789,  5.78947368,  6.31578947,  6.84210526,  7.36842105,
#       7.89473684,  8.42105263,  8.94736842,  9.47368421, 10.        ])
print(a)



#numpy basic operations
a= np.array([1,2,3])
b= np.array([4,5,6]) #Dimensions must be same for basic math operations.

print(a+b) #[5 7 9]
print(a-b) #[-3 -3 -3]
print(a**2) #[1 4 9]





#indexing stars from 0.
a= np.array([1,2,3,4,5,6,7]) #vector -> dimension 1






liste = [1,2,3,4]

array = np.array([5,6,7,8])
#diffrence between list and numpy array.
print(liste)
print(array)




array = np.array(liste)
print(array)


liste2 = list(array) #conversion from array to list.
print(liste2)



a = np.array([1,2,3])
print(a)



#Reference Adressing
b=a

c=a

b[0]=5 #All array a, b ,c 's 0 index will be updated as a 5 because b points the same adress as b and c.


print(a,b,c)






d = a.copy() #it is allocated new memory area with copy function so values are not be affected when it is changed





a= np.array([1,2,3,4,5,6,7]) #vector -> dimension 1
print(a)




print(a[0]) #accesing first column.





print(a[0:4]) #accesing first four column.




reverse_array = a[::-1] #getting columns in a reverse manner.
print(reverse_array)





b = np.array([[1,2,3,4,5],[6,7,8,9,10]]) 
print(b)




#indexs start from 0!

print(b[1,1]) #accessing 2nd row and 2nd column.

print(b[:,1]) #accessing all rows and 2nd column. [2 7]

print(b[1,:]) #accesing all column in the 2nd row. [ 6  7  8  9 10]

print(b[1,1:4]) #accesing 2nd 3rd 4th columns in the 2nd row. [7 8 9]

print(b[-1,:]) #accesing all columns in the last row. [ 6  7  8  9 10]

print(b[:,-1]) #accesing all rows in the last column. [ 5 10]





array = np.array([[1,2,3],[4,5,6],[7,8,9]]) 
print(array)




a = array.ravel() #Size was reducted 1x1 vector with ravel function.
print(a)




array2 = a.reshape(3,3)
print(array2)





array_transpose = array2.T #Transpose of array.
                           #array([[1, 4, 7],
                                  #[2, 5, 8],
                                  #[3, 6, 9]])
print(array_transpose)






array3 = np.array([[1,2],[3,4],[5,6]])
print(array3)




array3.resize((2,3)) #Size changes made with resize are permanent, they are saved in the same array.
#Changes with reshape are not permanent. the change made must be saved in another variable


print(array3)




######################################--------RECAP NUMPY---------#####################################








######################################--------RECAP PANDAS---------#####################################




t = (1,2,3,4)

t[1:3]




#importing the Pandas library.

import pandas as pd

dictionary = {"name":["ali","veli","zübeyde","ahmet","kubra","can"],
             "age":[12,34,56,78,None,12],
             "note":[123,456,78,87654,None,89]}

dataframe1 = pd.DataFrame(dictionary) 
print(dataframe1)




import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("") 

# Print the contents of the DataFrame
print(df)




head = dataframe1.head() 
print(head)



tail = dataframe1.tail()
print(tail)





print(dataframe1.columns) # getting names of columns





print(dataframe1.info()) #getting info about data frame





#print(dataframe1.dtypes) getting data type of each columns




print(dataframe1.describe()) 







print(dataframe1["name"]) 
print(dataframe1.loc[:, "age"])                     
dataframe1["yeni_future"] = [1,2,3,4,5,6]
print(dataframe1.loc[:3,"age"]) 
print(dataframe1.loc[:3, "name":"note"])                 #getting first 3 rows of till note column
print(dataframe1.loc[::-1])                             #printing reverse





print(dataframe1.yeni_future) 




print(dataframe1.loc[:, "age"]) # alternative of getting a column






print(dataframe1.loc[:3,"age"]) 






print(dataframe1.loc[:3, "name":"note"]) 






print(dataframe1.loc[:3, ["name","note"]]) #getting name and note columns till 3 rows






print(dataframe1.loc[::-1]) #printed reverse





print(dataframe1.loc[:,:"age"]) #all rows printed till age column including age.




print(dataframe1.iloc[:,[2]]) #all rows in the 2nd index was printed.



filtre1 = dataframe1.age>10
dataframe1["bool"]= filtre1
print(dataframe1.loc[:,["age","bool"]])





type(filtre1) 







filtrelenmis_data= dataframe1[filtre1] #filtered dataframe1,printing conditions was met
print(filtrelenmis_data)








filtre2 = dataframe1.note>100
filtrelenmis_data2 = dataframe1[filtre2&filtre1] #using two filter together
print(filtrelenmis_data2)







dataframe1[dataframe1.age>20] #using filter







import numpy as np
ortalama = dataframe1.note.mean() #finding mean with pandas
print(ortalama)
ortlama_np= np.mean(dataframe1.note) #finding mean with numpy
print(ortlama_np)





dataframe1.dropna(inplace=True) #deleted Nan values with dropna funct 







print(dataframe1.note.mean()) 

#printing above or under mean cheching it
dataframe1["ortalama"]= ["ortalamanın altında" if dataframe1.note.mean()>each else "ortalamanın üstünde" for each in dataframe1.note]
dataframe1






dataframe1.columns = [each.upper() for each in dataframe1.columns]
#dataframe1.columns






dataframe1["yeni2_future"]=[1,1,1,1,1]
dataframe1.columns = [each.split('_')[0]+" "+each.split('_')[1] if len(each.split('_'))>1 else each for each in dataframe1.columns]
#dataframe1




dataframe1.columns = [ each.split(" ")[0]+"_"+each.split(" ")[1] if len(each.split(" "))>1 else each for each in dataframe1.columns]
#dataframe1






dataframe1.drop(["yeni2_future","YENI_FUTURE"],axis=1,inplace=True)
#dataframe1



#verticle concatenation with 2 data frame
data1 = dataframe1.head()     
data2 = dataframe1.tail()
data_concat = pd.concat([data1,data2],axis=0)
#data_concat





#horizontal concatenation
data_contact2 = pd.concat([data1,data2],axis=1) #ıf axis= 1 it means column concatenation axis=0-> row concatenation
#data_contact2







dataframe1["buyuk_yas"] = [each*2 for each in dataframe1.AGE]
#dataframe1





def mlt(yas):
    return yas*2
dataframe1["apply_metodu"] = dataframe1.AGE.apply(mlt)
#dataframe1


######################################--------RECAP PANDAS---------#####################################













######################################--------RECAP MATPLOTLIB---------#####################################
import matplotlib.pyplot as plt 
import numpy as np

# Data
x = np.arange(0, 10, 0.1)
# [0.  0.1 0.2 0.3 0.4 0.5 ... 9.5 9.6 9.7 9.8 9.9]
y = np.sin(x)

# Graph creation
plt.plot(x, y)

# Determine chart properties
plt.title("Sinüs Grafiği")
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")

# Show graphics

plt.show()




import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = np.random.randint(50, 150, 50)

# Graph creation
plt.scatter(x, y, c=colors, s=sizes)

# Determine chart properties
plt.title("Point Chart")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Show graphics
plt.show()






import matplotlib.pyplot as plt
import numpy as np

# Data
data = np.random.randn(1000)

# Graph creation
plt.hist(data, bins=30)

# Determine chart properties
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")

# Show graphics
plt.show()





import matplotlib.pyplot as plt
import numpy as np

# Data
x = ['A', 'B', 'C', 'D', 'E']
y = np.random.randint(1, 10, 5)

# Graph creation
plt.bar(x, y)

# Determine chart properties
plt.title("Column Chart")
plt.xlabel("Categories")
plt.ylabel("Values")

# Show graphics
plt.show()




import matplotlib.pyplot as plt

# Data
sizes = [30, 25, 15, 10, 5, 5]

# Graph creation
plt.pie(sizes)

# Determine chart properties
plt.title("Pie Chart")

# Show graphics
plt.show()




import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Data
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)

# 3D graphics creation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plotting graphics
ax.plot_surface(x, y, z)

# Determine chart properties
ax.set_title("3D Graphics")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

# Show
plt.show()







######################################--------RECAP MATPLOTLIB---------#####################################

