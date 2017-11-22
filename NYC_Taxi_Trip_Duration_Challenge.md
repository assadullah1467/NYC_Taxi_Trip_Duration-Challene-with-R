---
title: "NYC_Taxi_Trip_Duration_Challenge"
author: "Assadullah_Samir"
date: "5 November 2017"
output:
    html_document:
    #  keep_md: yes
      code_folding: hide
      fig_caption: yes
      fig_height: 4.5
      highlight: tango
      keep_md: yes
      number_sections: yes
      theme: cosmo
      toc: yes


---


# Introduction
This is a comprehensive Exploratory Data Analysis for the New York City Taxi Trip Duration competition with tidy R and ggplot2.
The goal of this playground challenge is to predict the duration of taxi rides in NYC based on features like trip coordinates or pickup date and time. The data comes in the shape of 1.5 million training observations (../input/train.csv) and 630k test observation (../input/test.csv). Each row contains one taxi trip.
In this notebook, we will first study and visualise the original data, engineer new features, and examine potential outliers. Then we add two external data sets on the NYC weather and on the theoretically fastest routes. We visualise and analyse the new features within these data sets and their impact on the target trip_duration values. Finally, we will make a brief excursion into viewing this challenge as a classification problem and finish this notebook with a simple XGBoost model that provides a basic prediction (final part under construction).


```r
library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation
library('alluvial') # visualisation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('lubridate') # date and time
library('geosphere') # geospatial locations
library('leaflet') # maps
library('ggmap') # maps
library('maps') # maps
library('xgboost') # modelling
library('caret') # modelling
library('plotly') #visualization
library ('gridExtra') # arrange plots
```
Load data


```r
train <- as.tibble(fread('C:/Users/assad/NYC_Kaggle/train.csv'))
```

File structure and content


```r
summary(train)
```

```
##       id              vendor_id     pickup_datetime    dropoff_datetime  
##  Length:1458644     Min.   :1.000   Length:1458644     Length:1458644    
##  Class :character   1st Qu.:1.000   Class :character   Class :character  
##  Mode  :character   Median :2.000   Mode  :character   Mode  :character  
##                     Mean   :1.535                                        
##                     3rd Qu.:2.000                                        
##                     Max.   :2.000                                        
##  passenger_count pickup_longitude  pickup_latitude dropoff_longitude
##  Min.   :0.000   Min.   :-121.93   Min.   :34.36   Min.   :-121.93  
##  1st Qu.:1.000   1st Qu.: -73.99   1st Qu.:40.74   1st Qu.: -73.99  
##  Median :1.000   Median : -73.98   Median :40.75   Median : -73.98  
##  Mean   :1.665   Mean   : -73.97   Mean   :40.75   Mean   : -73.97  
##  3rd Qu.:2.000   3rd Qu.: -73.97   3rd Qu.:40.77   3rd Qu.: -73.96  
##  Max.   :9.000   Max.   : -61.34   Max.   :51.88   Max.   : -61.34  
##  dropoff_latitude store_and_fwd_flag trip_duration    
##  Min.   :32.18    Length:1458644     Min.   :      1  
##  1st Qu.:40.74    Class :character   1st Qu.:    397  
##  Median :40.75    Mode  :character   Median :    662  
##  Mean   :40.75                       Mean   :    959  
##  3rd Qu.:40.77                       3rd Qu.:   1075  
##  Max.   :43.92                       Max.   :3526282
```

```r
glimpse(train)
```

```
## Observations: 1,458,644
## Variables: 11
## $ id                 <chr> "id2875421", "id2377394", "id3858529", "id3...
## $ vendor_id          <int> 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2...
## $ pickup_datetime    <chr> "2016-03-14 17:24:55", "2016-06-12 00:43:35...
## $ dropoff_datetime   <chr> "2016-03-14 17:32:30", "2016-06-12 00:54:38...
## $ passenger_count    <int> 1, 1, 1, 1, 1, 6, 4, 1, 1, 1, 1, 4, 2, 1, 1...
## $ pickup_longitude   <dbl> -73.98215, -73.98042, -73.97903, -74.01004,...
## $ pickup_latitude    <dbl> 40.76794, 40.73856, 40.76394, 40.71997, 40....
## $ dropoff_longitude  <dbl> -73.96463, -73.99948, -74.00533, -74.01227,...
## $ dropoff_latitude   <dbl> 40.76560, 40.73115, 40.71009, 40.70672, 40....
## $ store_and_fwd_flag <chr> "N", "N", "N", "N", "N", "N", "N", "N", "N"...
## $ trip_duration      <int> 455, 663, 2124, 429, 435, 443, 341, 1551, 2...
```

Reformating Data to create hour and month of the trips


```r
train <- train %>%
  mutate(pickup_datetime = ymd_hms(pickup_datetime),
         dropoff_datetime = ymd_hms(dropoff_datetime),
         vendor_id = factor(vendor_id),
         passenger_count = factor(passenger_count))
## Create hpick and Month col
train<-train %>%
  mutate(hpick = hour(pickup_datetime),
         Month = factor(month(pickup_datetime, label = TRUE)))
head(train)
```

```
## # A tibble: 6 x 13
##          id vendor_id     pickup_datetime    dropoff_datetime
##       <chr>    <fctr>              <dttm>              <dttm>
## 1 id2875421         2 2016-03-14 17:24:55 2016-03-14 17:32:30
## 2 id2377394         1 2016-06-12 00:43:35 2016-06-12 00:54:38
## 3 id3858529         2 2016-01-19 11:35:24 2016-01-19 12:10:48
## 4 id3504673         2 2016-04-06 19:32:31 2016-04-06 19:39:40
## 5 id2181028         2 2016-03-26 13:30:55 2016-03-26 13:38:10
## 6 id0801584         2 2016-01-30 22:01:40 2016-01-30 22:09:03
## # ... with 9 more variables: passenger_count <fctr>,
## #   pickup_longitude <dbl>, pickup_latitude <dbl>,
## #   dropoff_longitude <dbl>, dropoff_latitude <dbl>,
## #   store_and_fwd_flag <chr>, trip_duration <int>, hpick <int>,
## #   Month <ord>
```

# Individual feature visualisations

Visualisations of feature distributions and their relations are key to understanding a data set, and they often open up new lines of inquiry. I always recommend to examine the data from as many different perspectives as possible to notice even subtle trends and correlations.

We start with a map of NYC and overlay a managable number of pickup coordinates to get a general overview of the locations and distances in question.


```r
set.seed(25)
foo <- train[sample(nrow(train), 80000), ]
nyc_map <- get_map(location = "new york", zoom = 10,maptype = "toner-2010",source = "stamen")
ggmap(nyc_map,extent = "device") +
  geom_point(aes(x = pickup_longitude, y = pickup_latitude,color="red"), data = foo, alpha = .5)
```

![](fig/unnamed-chunk-6-1.png)<!-- -->

```r
ggmap(nyc_map,extent = "panel") +
  geom_density2d(aes(x = pickup_longitude, y = pickup_latitude), data = train)+
  stat_density2d(data = train, aes(x = pickup_longitude, y = pickup_latitude, fill = ..level.., alpha = ..level..),size = 0.01, bins = 16, geom = 'polygon')
```

![](fig/unnamed-chunk-7-1.png)<!-- -->
It turns out that almost all of our trips were in fact taking place in Manhattan only. Another notable hot-spot is JFK airport towards the south-east of the city.

The map gives us an idea what some of the our distributions could look like. Let's start with plotting the target feature *trip\_duration*:

## Distribution of log of the trip_duartions


```r
ggplot(train,aes(trip_duration,fill=vendor_id))+geom_histogram(bins = 100)+scale_x_log10()
```

![](fig/unnamed-chunk-8-1.png)<!-- -->

Over the year, the distributions of *pickup\_datetime* and *dropoff\_datetime* look like this:


```r
ggplot(train,aes(pickup_datetime)) +geom_histogram(fill = "red", bins = 120) +  labs(x = "Pickup dates")
```

![](fig/unnamed-chunk-9-1.png)<!-- -->

```r
ggplot(train, aes(dropoff_datetime)) +geom_histogram(fill = "blue", bins = 120) +  labs(x = "Dropoff dates")
```

![](fig/unnamed-chunk-10-1.png)<!-- -->
## Binning of trip_duration
Duartion of the trips range from 1 sec to over 30 days but most of the journeys are under 1 hour. So binning the duration in 5 minute intervals, upto 1 hour, as trip_length-

```r
## Function for trip_duration categorization
duration.cat <- function(x, lower = 0, upper, by = 300,
                    sep = "-", above.char = "+") {

  labs <- c(paste(seq(lower, upper - by, by = by),
                  seq(lower + by - 1, upper - 1, by = by),
                  sep = sep),
            paste(upper, above.char, sep = ""))

  cut(floor(x), breaks = c(seq(lower, upper, by = by), Inf),
      right = FALSE, labels = labs)
}
## Binning using the function
train<-train %>%
  mutate(trip_length = duration.cat(trip_duration,upper=3600))
```
Distribution of trip_length

```r
ggplot(train,aes(trip_length,fill=factor(vendor_id)))+geom_bar()+theme(axis.text.x = element_text(angle = 90, hjust = 1))
```

![](fig/unnamed-chunk-12-1.png)<!-- -->
Average duration of the trips at diffrent month and hours:


```r
avg_duration=mean(train$trip_duration)
grp_df<-train%>%
          group_by(Month,hpick,vendor_id)%>%
          summarise(avg_dur=mean(trip_duration) )
ggplot(grp_df,aes(hpick,avg_dur,color=factor(vendor_id)))+geom_line()+facet_grid(Month~.)+
  geom_hline(yintercept = avg_duration, color="yellow")
```

![](fig/unnamed-chunk-13-1.png)<!-- -->
Number of trips at different time of the day:

```r
ggplot(train,aes(hpick,fill=factor(vendor_id)))+geom_histogram(bins = 24)
```

![](fig/unnamed-chunk-14-1.png)<!-- -->

Vendor 2 taxis have consistently longer average *trip_duration accross most hours in all the months. Trips taken during 10am-5pm are longer than other time for both the vendors.
