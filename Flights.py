source_city = "chi"
destination_city = "phx"
depart_date = "04/22/2016"
return_date = "04/24/2016"

expedia_str = "https://www.expedia.com/Flights-Search?trip=roundtrip&leg1=from:" \
              "{0},to:{1},departure:{2}TANYT&leg2=from:{1},to:{0},departure:{3}TANYT&" \
              "passengers=children:0,adults:1,seniors:0,infantinlap:Y&mode=search".format(
              	source_city, destination_city, depart_date, return_date)


