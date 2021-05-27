#!/usr/bin/env Rscript
library(argparser)
library(dplyr)
library(ggplot2)
library(tools) # for file_path_sans_ext()

plotPredictions <- function(file_path, out_dir, auto_bounds=FALSE) {
  df <- read.csv(file_path)

  # Rename actual and prediction columns to appease ggplot.
  for (col in colnames(df)) {
      if (grepl('actual$', col)) {
        df['actual'] <- df[col]
      }
      else if (grepl('prediction$', col)) {
        df['prediction'] <- df[col]
      }
  }

  if (auto_bounds == TRUE) {
    min_val=min(min(df['actual']), min(df['prediction']))
    max_val=max(max(df['actual']), max(df['prediction']))
    axis_min = min_val - 0.05*(max_val - min_val)
    axis_max = max_val + 0.05*(max_val - min_val)
  } else {
    axis_min=0
    axis_max=100
  }
  p <- df %>% ggplot(aes(x=actual, y=prediction)) + 
    geom_point(alpha = 0.8) +
    geom_smooth(method = 'lm', se = FALSE) + 
    ggpubr::stat_cor(method = 'pearson') +
    xlim(axis_min, axis_max) +
    ylim(axis_min, axis_max) +
    #scale_colour_viridis_c()
    geom_abline(slope=1, intercept=0) + 
    theme(text=element_text(size=24))
  
  # Save plot
  if (is.na(out_dir)) {
    out_dir = dirname(file_path)
  }
  out_name_without_extension = file_path_sans_ext(basename(file_path))
  out_name = paste0(out_name_without_extension, '.png')
  out_path = file.path(out_dir, out_name)
  
  aspect_ratio <- 1
  height <- 8
  width <- height*aspect_ratio
  
  ggsave(out_path, height=height, width=width)
}

description <- 'Plot predicted versus actual.'
parser <- arg_parser(description)
parser <- add_argument(parser, 'file_path', help='path to CSV file with actual and predicted values')
parser <- add_argument(parser, '--out_dir', short='-o', help='output directory. Default: save to same directory as input CSV file.', default=NULL)
parser <- add_argument(parser, '--auto_bounds', short='-a', help='automatically determine axis bounds for figures.', flag=TRUE)
args <- parse_args(parser)

#plotPredictions(file_path)
#print('Args raw.')
#print(args) # first three things in args are junk; get rid of them?

args[[1]] <- NULL
args[['help']] <- NULL
args[['opts']] <- NULL

#print(args)
#print('Args with junk removed.')
#print(args) # first three things in args are junk; get rid of them?

do.call(plotPredictions, args)