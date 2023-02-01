#!/usr/bin/env Rscript
library(argparser)
suppressPackageStartupMessages(library(dplyr))
library(ggplot2)
library(blandr)
library(tools) # for file_path_sans_ext()

plotPredictions <- function(file_path, out_dir, suffix, x_min, x_max, y_min, y_max, label_pos_x_rel=0.2, u1=FALSE, u2=FALSE, u3=FALSE, label_errors=FALSE, aspect_ratio=1, title=NA, size_multiplier=1, color_lines=FALSE) {
  df <- read.csv(file_path)

  # Try to determine what data is to automatically generate axis labels.
  is_lvot = ('lvot_actual' %in% colnames(df))
  is_vti = ('vti_actual' %in% colnames(df))
  is_co = ('co_actual' %in% colnames(df))
  is_ci = ('ci_actual' %in% colnames(df))
  
  # Rename actual and prediction columns to appease ggplot.
  for (col in colnames(df)) {
      if (grepl('actual$', col)) {
        df['actual'] <- df[col]
      }
      else if (grepl('predicted$', col) || grepl('prediction$', col)) {
        df['prediction'] <- df[col]
      }
  }

  df[,'mean'] = (df['prediction'] + df['actual'])/2
  df[,'difference'] = df['prediction'] - df['actual']

  # Perform Ben Shapiro test to determine if the distribution of the differences are significantly different from normal.
  n_result <- shapiro.test(df$difference)
  #print('Shapiro-Wilks test for normality:')
  print(n_result)
  
  # Perform a one-sample t-test to determine if the bias is significantly different from 0.
  t_result <- t.test(df$difference, mu=0, alternative='two.sided')
  #print('One-sided t-test of differences:')
  print(t_result)
  
  # Set the axis boundaries
  x_min_val=min(df['mean'])
  x_max_val=max(df['mean'])
  y_min_val=min(df['difference'])
  y_max_val=max(df['difference'])
  avg = mean(df[['difference']])
  
  # Set axis limits.
  if (is.na(x_min)) {
    x_min = x_min_val - 0.05*(x_max_val - x_min_val)
  }
  if (is.na(x_max)) {
    x_max = x_max_val + 0.05*(x_max_val - x_min_val)
  }
  if (is.na(y_min)) {
    y_min = y_min_val - 0.05*(y_max_val - y_min_val)
  }
  if (is.na(y_max)) {
    y_max = y_max_val + 0.05*(y_max_val - y_min_val)
  }
  
  # Manually set the x position of the error bars.
  error_bar_x = x_min + (x_max - x_min) * label_pos_x_rel
  label_x_offset_rel = 0.015
  label_x_offset = (x_max-x_min) * label_x_offset_rel
  label_y_offset_rel = 0.02
  label_y_offset = (y_max - y_min) * label_y_offset_rel
  if (avg<0) {
     label_y_offset <- -label_y_offset
  }
  
  ## Get stats from Bland-Altman analysis
  blstats <- blandr.statistics(df$prediction, df$actual, sig.level=0.95) # sig.level is for confidence intervals; limit of agreement is always 95%
  print(blstats)
  
  ## Create Bland-Altman plot
  # Set axis labels.
  y_label = "predicted - actual"
  x_label = "mean of predicted and actual"

  if (!is.na(suffix)) {
    y_label <- paste0(y_label, " ", suffix)
    x_label <- paste0(x_label, " ", suffix)
  } else if (is_lvot == TRUE) {
    y_label = bquote('predicted - actual LVOT diameter (mm)')
    x_label = bquote(atop('mean of predicted and actual', 'LVOT diameter (mm)'))
  } else if (is_vti == TRUE) {
    y_label = bquote('predicted - actual VTI (cm)')
    x_label = bquote(atop('mean of predicted and actual', 'VTI (cm)'))
  } else if (is_co == TRUE) {
    y_label = bquote('predicted - actual CO (L/min)')
    x_label = bquote(atop('mean of predicted and actual', 'CO (L/min)'))
  } else if (is_ci == TRUE) {
    y_label = bquote('predicted - actual CI (L/min/'*m^2*')')
    x_label = bquote(atop('mean of predicted and actual', 'CI (L/min/'*m^2*')'))
  }
  

  
  text_size <- 16*size_multiplier
  axis_text_size <- 14*size_multiplier
  annotation_text_size <-  5*size_multiplier
  point_size <- 2*size_multiplier
  
  error_bar_width = (x_max_val - x_min_val)*0.06
  
  p <- blandr.draw(df$prediction, df$actual, ciDisplay=FALSE, ciShading=FALSE, plotTitle=NULL) +
    geom_point(size=point_size, color='blue') +
    theme(text=element_text(size=text_size, color='black'),
          axis.text=element_text(size=axis_text_size, color='black'),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=1)) +
    
    coord_cartesian(xlim = c(x_min, x_max), ylim = c(y_min, y_max)) +
    
    # Set aspect ratio
    #coord_fixed(ratio=1.5) +
     
    labs(y=y_label, x=x_label) +

    # 95% CI for bias
    geom_errorbar(mapping=aes(x=error_bar_x, ymin=blstats$biasLowerCI, ymax=blstats$biasUpperCI), width=error_bar_width) + 
    
    
    # 95% CI for upper limit of agreement
    geom_errorbar(mapping=aes(x=error_bar_x, ymin=blstats$upperLOA_lowerCI, ymax=blstats$upperLOA_upperCI), width=error_bar_width) +
    
    
    # 95% CI for lower limit of agreement
    geom_errorbar(mapping=aes(x=error_bar_x, ymin=blstats$lowerLOA_lowerCI, ymax=blstats$lowerLOA_upperCI), width=error_bar_width)

  ## Label the bias and limit of agreement
  # Set positions at which to place the labels.
  if (u1 == TRUE) {
    upperLOA_label_y = blstats$upperLOA - size_multiplier*label_y_offset/aspect_ratio
  }
  else {
    upperLOA_label_y = blstats$upperLOA + size_multiplier*label_y_offset/aspect_ratio
  }
  upperLOA_label_x = error_bar_x+label_x_offset

  if (u2 == TRUE) {
    bias_label_y = blstats$bias - size_multiplier*label_y_offset/aspect_ratio
  }
  else {
    bias_label_y = blstats$bias + size_multiplier*label_y_offset/aspect_ratio
  }
  bias_label_x = error_bar_x+label_x_offset*aspect_ratio

  if (u3 == TRUE) {
    lowerLOA_label_y = blstats$lowerLOA - size_multiplier*label_y_offset/aspect_ratio
  }
  else {
    lowerLOA_label_y = blstats$lowerLOA + size_multiplier*label_y_offset/aspect_ratio
  }
  lowerLOA_label_x = error_bar_x+label_x_offset*aspect_ratio

  if (label_errors=="none")
  { print('doing nothing') }
  else if (label_errors=="long")
  {
    p <- p +
      annotate("text", x=upperLOA_label_x, y=upperLOA_label_y, label=paste("mean difference + 1.96SD =", sprintf("%.2f",blstats$upperLOA)), hjust=0, size=annotation_text_size) +
      annotate("text", x=bias_label_x, y=bias_label_y, label=paste("mean difference =", sprintf("%.2f",blstats$bias)), hjust=0, size=annotation_text_size) +
      annotate("text", x=lowerLOA_label_x, y=lowerLOA_label_y, label=paste("mean difference - 1.96SD =", sprintf("%.2f",blstats$lowerLOA)), hjust=0, size=annotation_text_size)
  }
  else if (label_errors=="short") {
      p <- p +
        annotate("text", x=upperLOA_label_x, y=upperLOA_label_y, label=paste("+1.96 SD =", sprintf("%.2f",blstats$upperLOA)), hjust=0, size=annotation_text_size) +
      	annotate("text", x=bias_label_x, y=bias_label_y, label=paste("mean =", sprintf("%.2f",blstats$bias)), hjust=0, size=annotation_text_size) +
      	annotate("text", x=lowerLOA_label_x, y=lowerLOA_label_y, label=paste("-1.96 SD =", sprintf("%.2f",blstats$lowerLOA)), hjust=0, size=annotation_text_size)
  }
  
  # Set title if one was specified.
  if (!is.na(title)) {
     p <- p + ggtitle(title)
  }
    
  # Overwrite the lines (bias and CI) with new lines. 
  if (color_lines == TRUE) {
    p <- p + 
      geom_hline(aes(yintercept=blstats$upperLOA), color="red", linetype="dashed") +
      geom_hline(aes(yintercept=blstats$bias), color="blue", linetype="dashed") +
      geom_hline(aes(yintercept=blstats$lowerLOA), color="red", linetype="dashed")
  }
  
  # Save plot
  if (is.na(out_dir)) {
    out_dir = dirname(file_path)
  }
  out_name_without_extension = file_path_sans_ext(basename(file_path))
  out_name = paste0(out_name_without_extension, '-BlandAltman.png')
  out_path = file.path(out_dir, out_name)
  
  height <- 8
  width <- height*aspect_ratio
  ggsave(out_path, height=height, width=width)
}

description <- 'Create Bland-Altman plot and print metrics'
parser <- arg_parser(description)
parser <- add_argument(parser, 'file_path', help='path to CSV file with actual and predicted values')
parser <- add_argument(parser, '--out_dir', short='-o', help='output directory. Default: save to same directory as input CSV file.', default=NULL)
parser <- add_argument(parser, '--suffix', short='-s', help='suffix for axis labels (e.g. "(cm)")', type="character")
parser <- add_argument(parser, '--x_min', short='-l', help='x-axis min (left limit)', type="numeric")
parser <- add_argument(parser, '--x_max', short='-r', help='x-axis max (right limit)', type="numeric")
parser <- add_argument(parser, '--y_min', short='-b', help='y-axis min (bottom limit)', type="numeric")
parser <- add_argument(parser, '--y_max', short='-t', help='y-axis max (top limit)', type="numeric")
parser <- add_argument(parser, '--label_pos_x_rel', short='-e', help='relative position of error bars along the x axis.', default=0.20, type="numeric")
parser <- add_argument(parser, '--u1', help='put the label for the upper limit of agreement under the line', flag=TRUE)
parser <- add_argument(parser, '--u2', help='put the label for the bias under the line', flag=TRUE)
parser <- add_argument(parser, '--u3', help='put the label for the lower limit of agreement under the line', flag=TRUE)
parser <- add_argument(parser, '--label_errors', help='label the error bars verbosely', type="character", default="none")
parser <- add_argument(parser, '--aspect_ratio', help='aspect ratio (x/y)', default=1, type='numeric')
#parser <- add_argument(parser, '--title', help='title text', default=NA)
parser <- add_argument(parser, '--color_lines', help='color the bias and CI lines (blue-solid and red-dashed)', flag=TRUE)
parser <- add_argument(parser, '--size_multiplier', short='-z', help='multiply font and point sizes by this number', default=1, type='numeric')
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