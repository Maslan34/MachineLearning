var1 =10  #declaration of integer value

var2=20

day="monday" #declaration of string value

var3=10.0 #declaration of float value



list_int=[1,2,3,4,5] #declaration of array
print(list_int) #printing an array



type(list_int)

value=list_int[0] #accesing an array element
print(value)


value2=list_int[-1] #accesing an array element
print(value2)

value3=list_int[0:2] #accesing an array elements
print(value3)



dir(list_int)



list_int.append(11) #adding  an element to array
print(list_int)


list_int.remove(3) #deleting   an element exist in array
print(list_int)

list_int.reverse() 
print(list_int)


list_int.sort()
print(list_int)


for each in range(1,11): #for each loop
    print(each)


list=[2,3,56,23,123,113,1341,13]

print("sum: "+str(sum(list))) #concatenation and using a function


print("min: "+str(min(list)))


minimum=1000000


for each in list:
    if(each<minimum):
        minimum=each
    else:
        continue

print("loop min: "+str(minimum))

i=0

while(i<4):   #while loop
    print(i)
    i=i+1
    
    
def area_circle(r,pi=3.14):  # declearation of function
    rValue=pi*r*r
    return rValue


print(area_circle(4))

def squareE(x):
    return x*x


sonuc=squareE(4)
print("area:"+str(sonuc))

sonuc2=lambda x: x*x   # declearation of lambda function
print("area:"+str(sonuc2(4)))



dictionary={"muharrem": 22,"aslan":100}  # declearation of dictionary(sets)
print(dictionary["aslan"])

print(dictionary.keys())

print(dictionary.values())

def get_dic():
    dic={"muharrem ": 44,"aslan":200}
    return dic


dic=get_dic()
print(dic)

keys=dictionary.keys()

if "muharrem" in keys: 
    print("true")
else:
    print("false")
    
if "muharremm" in keys:
    print("true")
else:
    print("false")

    
    