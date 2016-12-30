#COPIED FROM WORD DOC AND CHANGED TO SUIT PYTHON 3.X
"""
Original data:

100010111010101000000010111011010010000000001011000001100000010000010010010111011000110010111000010101011000000010101000001010100000000000000011111001110110

Delete all zeros at the beginning. Find all sequences of three or more zeros and replace them with sequences of n mod 3 zeros, followed by a comma. Note for later: if a segment of the input (offset by commas / periods) is empty, ignore it (produce no output).

1,101110101010,1011101101001,101100,11,100,10010010111011,1100101110,101010110,1010100,10101,11111001110110.

THESE ARE MULTIPLE STEPS, SOME PRETTY COMPLICATED
Starting from the left, divide the string into segments. Each segment must begin with 1 and be between 1 and 5 characters long (inclusive). If a segment is all ones, its length is cut off at 4 (but be sure that the segment 11110 is still possible). If a comma is reached, the segment ends (no comma is added). If a period is reached, the segment and the text end (no comma is added). If a segment of length 5 is created and the next bit is zero, the segment should be cut off at the bit before the last 1 in it, so that this one may serve as the beginning of the next segment (which will, under these conditions, have a second bit of zero).

1,1011,1010,1010,1011,10110,1001,101,100,11,100,100,10010,11101,1,1100,1011,10,1010,10110,1010,100,10101,1111,10011,10110.

Translate the segments into the 26 letters using the key:
a=1 b=10 c=11 d=100 e=101 f=110 g=111 h=1001 i=1010 j=1011 k=1100 l=1101 m=1110 n=1111 o=10010 p=10011 q=10100 r=10101 s=10110 t=10111 u=11001 v=11010 w=11011 x=11100 y=11101 z=11110

AJIIJSHEDCDDOYAKJBISIDRNPS

Done!

Put all of this code in a function of decode(hat) or something. Then make a while loop with an incremented counter (up to some high number, trials), in which an input of length hat_length is generated at random and then translated. Add each output to a long string (as long as you’re not looking at billions of characters, it should be fine). After all the simulations are done (when the while loop terminates), count the occurrences of each letter. Sort them by number or proportion of occurrences, then print, e.g., “S: 7.51%”, in descending order of frequency. All of this simulation process should be in a function of something like simulate(hat, hat_length, trials). 

Once you have sufficient data as to the relative frequency of the letters under this system (ideally in the positive infinite limit of both hat_length and trials), consider reassigning the values in the key such that the frequencies correspond as closely as possible to those of the actual letters in English (however, keep it one-to-one; no combining; no splitting).
"""

#Code so far (Python):

end = False
hat = input("Please enter the binary string representing the data collected. ")
for u in range(len(hat)):
    if hat[u] != "0" and hat[u] != "1":
        print("Input must be a binary string. Error occurred at index %s." % u)
        end = True
        #maybe have a more sophisticated error message that lists all the error spots, but this doesn’t really matter
    else:
        pass
    
if end == False:
    hat_closed = hat + "."
    #print("hat_closed = %s" % hat_closed)
    
if end == False:
    lzc = 0 #leading zero count
    for u in range(len(hat_closed)):
        if hat_closed[u] == "0":
            lzc += 1
        elif hat_closed[u] == "1":
            break
    if lzc >= len(hat_closed) - 1:
        end = True
    hat_shaved = hat_closed[lzc:]
    #print("hat_shaved = %s" % hat_shaved)

if end == False:
    sozrs = {} #starts of zero runs
    for u in range(len(hat_shaved)):
        if hat_shaved[u] == "0" and hat_shaved[u+1] == "0" and hat_shaved[u+2] == "0":
            #store the index for reference, find out how many zeros are in the run (use a dict of start:length to match these), then after the for loop, build the new string by checking whether indices are in the dict for starting zero runs
            rzc = 0 #running zero count
            for e in range(u, len(hat_shaved)):
                if hat_shaved[e] == "0":
                    rzc += 1
                elif hat_shaved[e] == "1":
                    break
            indices_in_zero_runs = [j + k for j in sorted(sozrs.keys()) for k in range(sozrs[j])]
            if u not in indices_in_zero_runs:
                sozrs[u] = rzc
    hat_snipped = ""
    indices_to_use = [u for u in range(len(hat_shaved))]
    for u in sorted(sozrs.keys()):
        for e in range(1, sozrs[u]):
            if u + e in indices_to_use:
                indices_to_use.remove(u + e)
    for u in indices_to_use:
        if u not in sozrs:
            hat_snipped = hat_snipped + hat_shaved[u]
        elif u in sozrs:
            hat_snipped = hat_snipped + ("0" * (sozrs[u] % 3)) + ","
    #print(indices_to_use)
    #print(sozrs)
    print("hat_snipped = %s" % hat_snipped)
    
if end == False:
    segstarts = [0]
    segs = {}
    segnum = 0
    for u in segstarts:
        seg = ""
        for e in range(5):
            if hat_snipped[u + e] == "," or hat_snipped[u + e] == ".":
                k = e
                break
        else:
            for i in range(5):
                if hat_snipped[u + 5 - i] == "0":
                    k = 3 - i # = 5 - i - 2
            else:
                k = 4
        for e in range(k + 1):
            if hat_snipped[u + e] == ",":
                segstarts.append(u + e + 1)
                break
            elif hat_snipped[u + e] == ".":
                end = True
                break
            elif seg == "1111" and hat_snipped[u + e] == "1":
                seg = "1111"
                segstarts.append(u + e)
                break
            elif e == k:
                seg = seg + hat_snipped[u + e]
                segstarts.append(u + e + 1)
            else:
                seg = seg + hat_snipped[u + e]
        segnum += 1
        segs[segnum] = seg
    print(segs)
    # needs work. see the output of segs for the input on the Word document. it gives segments that start with zero. IF ERROR, I PUT SPACES IN FRONT OF THIS IN WORD.
