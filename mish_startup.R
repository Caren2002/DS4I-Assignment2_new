library(keras3)
library(imager)
library(keras3)
use_backend("tensorflow")



data <- readRDS("handwriting.rds")
dim(data$images) # 25000 x 28 x 28 x 1

# 25000 images
# dimension of each image is 28 x 28
# 1 value per pixel (monochrome)


summary(data$metadata)
str(data)

## column 2 is person id
# length(unique(data$metadata[1:1250,2]))   # 1
# length(unique(data$metadata[1:1251,2]))   # 2


## column 3 is digit
# length(unique(data$metadata[1:250,3]))    # 1
# length(unique(data$metadata[1:251,3]))    # 2


## column 3 is session
# length(unique(data$metadata[1:50,4]))    # 1
# length(unique(data$metadata[1:51,4]))    # 2

# visualizing the first 5 images
par(mfrow = c(1, 5))
for(i in 1:5){plot(as.cimg(t(data$images[i,,,])))}



















# Check dimensions
dim(data$images)

# Check metadata structure
str(data$metadata)

# Check for missing values
colSums(is.na(data$metadata))

# Check image pixel range
range(data$images)

# Number of unique writers, digits and sessions
length(unique(data$metadata$person_id))
length(unique(data$metadata$digit))
length(unique(data$metadata$session))

# Number of observations for each writer
table(data$metadata$person_id)

# Number of observations for each digit
table(data$metadata$digit)

# Number of observations for each session
table(data$metadata$session)


table(data$metadata$person_id, data$metadata$digit, data$metadata$session)



plot_digit <- function(i) {
  
  img <- data$images[i, , , 1]
  image(
    t(img[nrow(img):1, ]),
    col = gray.colors(256),
    axes = FALSE,
    useRaster = TRUE)
  box()
}




# Examples across writers ------
# Select digit 2 from session 1
idx <- which(data$metadata$digit == 2 & data$metadata$session == 1)

# Select one image from each writer
idx <- sapply(1:20, function(p) {
  idx[data$metadata$person_id[idx] == p][1]
})

# Plot
par(mfrow = c(4, 5), mar = c(1, 1, 2, 1))

for (i in idx) {
  
  img <- data$images[i, , , 1]
  
  image(
    t(img[nrow(img):1, ]),
    col = gray.colors(256),
    axes = FALSE,
    useRaster = TRUE)
  
  title(paste0("Writer ", data$metadata$person_id[i]))
}








# Examples across digits -----------
# Writer 1, session 1
idx <- sapply(0:4, function(d) {
  which(data$metadata$person_id == 1 & data$metadata$session == 1 & 
          data$metadata$digit == d)[1]
})

par(mfrow = c(1, 5), mar = c(12, 1, 12, 1))
#par(mfrow = c(1, 5))
for (i in idx) {
  
  img <- data$images[i, , , 1]
  
  image(
    t(img[nrow(img):1, ]),
    col = gray.colors(256),
    axes = FALSE,
    useRaster = TRUE)
  
  title(paste0("Digit ", data$metadata$digit[i]))
}




# Examples of a writer amongst sesssions (writer 1, digit 2)
idx <- sapply(1:5, function(s) {
  which(data$metadata$person_id == 1 & data$metadata$digit == 2 & 
          data$metadata$session == s)[1]
})

par(mfrow = c(1, 5),mar = c(12, 1, 12, 1))

for (i in idx) {
  
  img <- data$images[i, , , 1]
  image(
    t(img[nrow(img):1, ]),
    col = gray.colors(256),
    axes = FALSE,
    useRaster = TRUE)
  
  title(paste0("Session ", data$metadata$session[i]))
}







# Examples of variation of a person within a particular session 
idx <- which(data$metadata$person_id == 1 & data$metadata$digit == 2 & 
               data$metadata$session == 1)

# Select 10 replicates
set.seed(8)
idx <- sample(idx, 10)

par(mfrow = c(2, 5),mar = c(1, 1, 3, 1))

for (i in idx) {
  
  img <- data$images[i, , , 1]
  image(
    t(img[nrow(img):1, ]),
    col = gray.colors(256),
    axes = FALSE,
    useRaster = TRUE)
  
  title(paste0("Rep. ", data$metadata$replicate[i]))
}









# Examples of writers x sessions
# Choose digit
'''
chosen_digit <- 2

# Select one replicate for each writer-session combination
idx <- matrix(NA, nrow = 5, ncol = 20)

for (s in 1:5) {
  for (p in 1:20) {
    
    idx[s, p] <- which(data$metadata$person_id == p & 
                         data$metadata$digit == chosen_digit & 
                         data$metadata$session == s)[1]
  }
}

par(mfrow = c(5, 20), mar = c(0.2, 0.2, 0.8, 0.2))

for (s in 1:5) {
  for (p in 1:20) {
    
    i <- idx[s, p]
    img <- data$images[i, , , 1]
    
    image(
      t(img[nrow(img):1, ]),
      col = gray.colors(256),
      axes = FALSE,
      useRaster = TRUE)
    
    if (s == 1) {
      title(paste0("W", p), cex.main = 0.6)
    }
  }
}
'''




par(mfrow = c(1,1))

##### technical details of data ---- can come in handy for justification of resulst*** 

# Mean pixel intensity for each image
mean_intensity <- apply(data$images, 1, mean)

# Add to metadata
data$metadata$mean_intensity <- mean_intensity

hist(data$metadata$mean_intensity, breaks = 30, 
     main = "Distribution of mean pixel intensity",
     xlab = "Mean pixel intensity")

# across digits
boxplot(mean_intensity ~ digit, data = data$metadata, xlab = "Digit", 
        ylab = "Mean pixel intensity", main = "Pixel intensity by digit")

# across sessions
boxplot(mean_intensity ~ session, data = data$metadata, 
        xlab = "Session", ylab = "Mean pixel intensity", 
        main = "Pixel intensity by session")

