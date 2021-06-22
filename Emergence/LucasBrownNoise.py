# original author: Lucas Brown
#Adv. GIS final project
#5/30/19

#random interpolated surface generator

import arcpy
import random
from arcpy import env
env.workspace=arcpy.GetParameterAsText(2)
env.overwriteOutput = True
import arcpy.sa

grid_size = int(arcpy.GetParameterAsText(0))
variability = int(arcpy.GetParameterAsText(1))
Output_loc = arcpy.GetParameterAsText(2)
R_output = arcpy.GetParameterAsText(3)
Pt_output_name = arcpy.GetParameterAsText(4)

arcpy.CheckOutExtension('Spatial')

out_path= Output_loc
if Pt_output_name=="":
    name='grid.shp'
else:
    name = Pt_output_name+'.shp'
type='POINT'
arcpy.CreateFeatureclass_management(out_path,name,type)

size=grid_size
# size=10
for i in range(0,size):     #using a double loop to create a point grid of the selected size
    for j in range(0,size):
        pt1 = arcpy.Point(i, j)
        cursor1 = arcpy.da.InsertCursor(name, ['SHAPE@'])
        cursor1.insertRow([pt1])
        del cursor1

field='z_value'
data_type='FLOAT'
                # a new field is necesary to hold the z-data because all interpolation methods in arc require a z-field
arcpy.AddField_management(name,field,data_type)

grid_list=[]

seed = random.uniform(0, 10)    #seed value
first=seed                      #needs a separate vari to hold value for later
count=0
vari=variability
# vari=2
for i in range(0,size):
    if i>=1:        #for first row in other columns: the first point needs to be based on the first of the column created before
        next2=random.uniform(first - vari, first + vari)
        grid_list.append(next2)
        first=next2
        seed=next2
        count = count + 1
    else:
        grid_list.append(seed)  #fisrt column and row: just add point
        count = count + 1
    for j in range (1,size):
        if i >= 1:  #for points after the first column and row: their values need to be based on point just created as well as the point in the column before
            next3 = random.uniform(((seed+grid_list[count-size])/2)-(.75*vari), ((seed+grid_list[count-size])/2) + (.75*vari))
            grid_list.append(next3)                 #variation is decreased in these points compared to the others due to fewer degrees of freedom
            seed=next3
            count=count+1
        else:
            next = random.uniform(seed - vari, seed + vari)   #for the first column: add points based on the point that came before.
            grid_list.append(next)                            #this keeps the raster from being too random and unable to interpolate properly
            seed=next
            count=count+1

update=arcpy.da.UpdateCursor(name,field)
list_spot=0
for i in update:
    i[0]=grid_list[list_spot]   #update cursor to add data from the list into the point file
    update.updateRow(i)
    list_spot=list_spot+1
del update

# R_output='grid3_test'
if R_output=="":
    raster= 'Surface.tif'       #interpolation with kriging
else:
    raster =  R_output + '.tif'
surface=arcpy.sa.Kriging(name,field,'SPHERICAL')
surface.save(raster)


arcpy.CheckInExtension('Spatial')
