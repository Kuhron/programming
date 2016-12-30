import time,random
 
items = [chr(i) for i in range(0x30a1, 0x30ff + 1)] # katakana
 
 
for i in range(1,11): # spaces and numbers
    items.append(str(i))
    items.append(" "*i)
 
def rain(row,column):         # rows,columns
     for i in range(row): #for every row
             s = ''      #new string
             for j in range(column): #for every column (or character)
                     ri = random.randrange(len(items)) #random index
                     s += items[ri]
             print(s)
             time.sleep(0.05)

rain(1000000, 10)
