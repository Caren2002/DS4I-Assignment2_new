library(keras3)
library(imager)
library(keras3)
library(pROC)
use_backend("tensorflow")

data <- readRDS("handwriting.rds")


# Scale pixel values from [0, 255] to [0, 1]
images <- data$images / 255

# rename for easy references later
person_id <- data$metadata$person_id
digit <- data$metadata$digit
session <- data$metadata$session





# sessions 1,2,3 as training data
train_ids <- which(data$metadata$session %in% c(1, 2, 3))

# Session 4 as validation
valid_ids <- which(data$metadata$session == 4)

# Session 5 as test
test_ids <- which(data$metadata$session == 5)

# verify sizes
length(train_ids)
length(valid_ids)
length(test_ids)






set.seed(25)

# output are the indices of the images in each pair
make_pairs <- function(ids, n, person_labels = person_id, digit_labels = digit) {
  
  # Select first image in each pair, allow each image to be in multiple pairs
  a <- sample(ids, n, replace = TRUE)
  
  # Select second image
  b <- vapply(a,
              function(i) {
                # Decide if same or diff writer (50% chance)
                same_writer <- runif(1) < 0.5
                
                if (same_writer) { 
                  
                  # find inds of same writer AND same digit 
                  pool <- ids[person_labels[ids] == person_labels[i] & 
                                digit_labels[ids] == digit_labels[i]]
                  
                } else {
                  
                  # find inds of different writer BUT same digit
                  pool <- ids[person_labels[ids] != person_labels[i] & 
                                digit_labels[ids] == digit_labels[i]]
                }
                
                # Do not pair an image with itself
                # extract all unique image indices excluding the ind of the first image
                pool <- setdiff(pool, i)
                
                # pick 1 to be the second image
                sample(pool, 1)
              },
              integer(1)
  )
  
  cbind(a, b)
}




train_pairs <- make_pairs(ids = train_ids, n = 12000)

valid_pairs <- make_pairs(ids = valid_ids, n = 4000)

test_pairs <- make_pairs(ids = test_ids, n = 4000)







make_batch <- function(pair_indices) {
  
  # y = 1 if same writer
  # y = 0 if different writer
  
  y <- matrix(
    as.integer(
      person_id[pair_indices[, 1]] ==
        person_id[pair_indices[, 2]]
    ),
    ncol = 1
  )
  
  list(
    left = images[pair_indices[, 1], , , , drop = FALSE], #first image
    right = images[pair_indices[, 2], , , , drop = FALSE], #second image
    y = y,                                                # same or diff writer
    digit = digit[pair_indices[, 1]],                   # digit in image
    session_left = session[pair_indices[, 1]],         #session of first image
    session_right = session[pair_indices[, 2]]          # session of second
  )
}


train <- make_batch(train_pairs)
#check all digits are same in each pair
all(data$metadata$digit[train_pairs[, 1]] == 
      data$metadata$digit[train_pairs[, 2]])


valid <- make_batch(valid_pairs)
#check for same digits
all(data$metadata$digit[valid_pairs[, 1]] ==
      data$metadata$digit[valid_pairs[, 2]])


test <- make_batch(test_pairs)
all(data$metadata$digit[test_pairs[, 1]] == 
      data$metadata$digit[test_pairs[, 2]])





# build CNN ----------------------------

## define dim of input (images are 28 x 28 pixels, and 1 implies grayscale)
image_input <- layer_input(
  shape = c(28, 28, 1))

# first convolutional layer
embedding <- image_input |>
  layer_conv_2d(
    filters = 16, # 16 filters, each of size 3x3
    kernel_size = 3,
    activation = "relu" # max(0,x) (output from 1st conv layer is 26x26x16)
  ) |>
  # 2x2 subregions, takes max (at end of CNN slides) (dim is now 13x13x16)
  layer_max_pooling_2d( 
    pool_size = 2
  ) |>
  # second convolutional layer (check if actually necessary)
  layer_conv_2d(
    filters = 32,
    kernel_size = 3, #(produces output of 11x11x32)
    activation = "relu"
  ) |>
  layer_global_average_pooling_2d() |>  # outputs a vec of length 32 (1 value
  # for each filter map)
  layer_dense(16) # embedding forced to be 16 units long


# packages all of the above together (similar to creating a function)
encoder <- keras_model(
  inputs = image_input,
  outputs = embedding
)







## Siamese model ----------------

# Characteristics of each image in the pair
left_input <- layer_input(shape = c(28, 28, 1))
right_input <- layer_input(shape = c(28, 28, 1))

# get encodings for both images by passing them through the same encoder
distance <- list(
  encoder(left_input),
  encoder(right_input)) |> 
  
  # calc euclidean distance between the 2 embeddings
  layer_lambda(
    f = function(z) {
      op_sqrt(op_sum(op_square(z[[1]] - z[[2]]),axis = 2, keepdims = TRUE) 
              + 1e-7)
    }
  )

# package everything together
siamese_model <- keras_model(
  inputs = list(left_input, right_input),
  outputs = distance)


#loss function (y = 1 means same writer) 
contrastive_loss <- function(y, distance) {
  op_mean(y * op_square(distance) + (1 - y) * op_square(op_maximum(
    1 - distance,0)))
}


siamese_model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3), loss = contrastive_loss)


#Train siamese model ----------------------------------
history <- siamese_model |> fit(
  
  x = list(
    train$left,
    train$right),
  
  y = train$y,
  
  validation_data = list(
    list(
      valid$left,
      valid$right), 
    valid$y),
  
  epochs = 5,
  
  batch_size = 64)


plot(history)


# Validation distances
valid_distance <- as.numeric(
  predict(siamese_model, list(valid$left, valid$right)))



valid_same <- as.logical(valid$y[, 1])




balanced_accuracy_at <- function(cut) {
  
  predicts_same <- valid_distance < cut
  same_rate <- mean(predicts_same[valid_same])
  different_rate <- mean(!predicts_same[!valid_same])
  (same_rate + different_rate) / 2
}






# to find appropriate threshold
threshold_candidates <- sort(unique(valid_distance)) # could try seq of values?

threshold_scores <- vapply(threshold_candidates,balanced_accuracy_at, numeric(1))

best_index <- which.max(threshold_scores)

threshold <- threshold_candidates[best_index]

threshold



# use best threshold
validation_result <- data.frame(threshold = threshold, balanced_accuracy = threshold_scores[best_index])

validation_result

######--------------------------------------------------------------------------
# create different pairs for test set
set.seed(25)

# All test images are from session 5
test_ids <- which(data$metadata$session == 5)

# Function to create controlled test pairs
make_controlled_pairs <- function(ids, n, pair_type) {
  
  pairs <- matrix(NA_integer_, nrow = n, ncol = 2)
  
  for (j in 1:n) {
    
    # Choose first image
    i <- sample(ids, 1)
    
    person_i <- person_id[i]
    digit_i <- digit[i]
    session_i <- session[i]
    
    if (pair_type == "same_writer_same_session") {
      
      pool <- ids[person_id[ids] == person_i & digit[ids] == digit_i 
                  & session[ids] == session_i & ids != i]
      
    } else if (pair_type == "same_writer_different_session") {
      
      # For cross-session pairs, use sessions 1-4
      comparison_ids <- which(person_id == person_i & digit == digit_i 
                              & session != session_i)
      
      pool <- comparison_ids[comparison_ids %in% 
                               c(which(session %in% c(1, 2, 3, 4)))]
      
    } else if (pair_type == "different_writer_same_session") {
      
      pool <- ids[person_id[ids] != person_i & digit[ids] == digit_i 
                  & session[ids] == session_i]
      
    } else if (pair_type == "different_writer_different_session") {
      
      comparison_ids <- which(person_id != person_i & digit == digit_i 
                              & session != session_i 
                              & session %in% c(1, 2, 3, 4))
      
      pool <- comparison_ids
      
    }
    
    pairs[j, ] <- c(i, sample(pool, 1))
  }
  
  pairs
}




# use functin to get different types of test pairs
n_per_type <- 1000

test_pairs_same_same <- make_controlled_pairs(test_ids, n_per_type, 
                                              "same_writer_same_session")

test_pairs_same_diff <- make_controlled_pairs(test_ids, n_per_type,
                                              "same_writer_different_session")

test_pairs_diff_same <- make_controlled_pairs(test_ids, n_per_type,
                                              "different_writer_same_session")

test_pairs_diff_diff <- make_controlled_pairs(test_ids, n_per_type, 
                                              "different_writer_different_session")


# combine
test_pairs <- rbind(test_pairs_same_same,test_pairs_same_diff,
                    test_pairs_diff_same,test_pairs_diff_diff)

dim(test_pairs)


test_pair_type <- c(
  rep("Same writer, same session", n_per_type),
  rep("Same writer, different session", n_per_type),
  rep("Different writer, same session", n_per_type),
  rep("Different writer, different session", n_per_type)
)

#table(test_pair_type)

test <- make_batch(test_pairs)
test$pair_type <- test_pair_type


# check pairs are of correct formats, combinations
table(test_pair_type, person_id[test_pairs[,1]] == person_id[test_pairs[,2]])

all(digit[test_pairs[,1]] == digit[test_pairs[,2]])

table(test_pair_type, session[test_pairs[,1]] == session[test_pairs[,2]])

######--------------------------------------------------------------------------

# final eval using tuned parameters and other final mod choices
test_distance <- as.numeric(predict(siamese_model,list(test$left,test$right)))

true_same <- as.logical(test$y[, 1])

predicted_same <- test_distance < threshold

## format results
session_results <- lapply(unique(test_pair_type), function(s) {
  
  keep <- test_pair_type == s
  
  true_s <- true_same[keep]
  pred_s <- predicted_same[keep]
  
  data.frame(
    pair_type = s,
    n = sum(keep),
    correct = sum(pred_s == true_s),
    accuracy = mean(pred_s == true_s))
})


session_results <- do.call(rbind, session_results)
session_results



# performance metrics for just same or different session
session_type <- ifelse(session[test_pairs[, 1]] == session[test_pairs[, 2]], 
                       "Same session", "Different session")


##
session_results2 <- lapply(unique(session_type), function(s) {
  
  keep <- session_type == s
  
  true_s <- true_same[keep]
  pred_s <- predicted_same[keep]
  
  sensitivity_s <- mean(pred_s[true_s])
  
  specificity_s <- mean(!pred_s[!true_s])
  
  data.frame(
    session_type = s,
    n = sum(keep),
    accuracy = mean(pred_s == true_s),
    sensitivity = sensitivity_s,
    specificity = specificity_s,
    balanced_accuracy = (sensitivity_s + specificity_s) / 2)
})

session_results2 <- do.call(rbind,session_results2)
session_results2





# Performance by digit

digit_results <- lapply(0:4, function(d) {
  
  keep <- test$digit == d
  
  true_d <- true_same[keep]
  pred_d <- predicted_same[keep]
  
  sensitivity_d <- mean(pred_d[true_d])
  
  specificity_d <- mean(!pred_d[!true_d])
  
  data.frame(
    digit = d,
    accuracy = mean(pred_d == true_d),
    sensitivity = sensitivity_d,
    specificity = specificity_d,
    balanced_accuracy = (sensitivity_d + specificity_d) / 2)
})

digit_results <- do.call(rbind, digit_results)
digit_results



# Performance metrics per writer

writer_results <- lapply(sort(unique(person_id)), function(w) {
  
  # Select pairs where the FIRST image belongs to writer w
  keep <- person_id[test_pairs[, 1]] == w
  
  true_w <- true_same[keep]
  pred_w <- predicted_same[keep]
  
  # Calculate metrics
  sensitivity_w <- mean(pred_w[true_w])
  specificity_w <- mean(!pred_w[!true_w])
  
  data.frame(
    writer = w,
    n = sum(keep),
    accuracy = mean(pred_w == true_w),
    sensitivity = sensitivity_w,
    specificity = specificity_w,
    balanced_accuracy = (sensitivity_w + specificity_w) / 2
  )
})

writer_results <- do.call(rbind, writer_results)

writer_results



# ROC AUC CURVE

roc_object <- roc(response = true_same, predictor = -test_distance)
auc_value <- auc(roc_object)
auc_value


plot(roc_object, main = "Siamese Network ROC Curve")




