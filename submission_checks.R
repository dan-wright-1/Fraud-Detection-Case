
expected_object_names <- c(
  'air',                
  'air_folds',
  'air_final_fit',
  'air_lr_corr_cv_fit', 
  'air_lr_corr_wkfl',   
  'air_lr_pca_cv_fit',  
  'air_lr_pca_wkfl',    
  'air_perf_summary',   
  'air_rec_corr',       
  'air_rec_pca',        
  'air_split',          
  'air_testing',        
  'air_training',       
  'air_xgb_corr_cv_fit',      
  'air_xgb_corr_wkfl',  
  'air_xgb_pca_cv_fit', 
  'air_xgb_pca_wkfl',   
  'best_parameters',    
  'final_lc_wkfl',      
  'fraud',              
  'fraud_fit_1',        
  'fraud_fit_2',        
  'fraud_fit_3',        
  'fraud_fit_4',        
  'fraud_fit_5',        
  'fraud_fit_6',        
  'fraud_metric_set',  
  'fraud_perf_summary', 
  'fraud_rec_1',        
  'fraud_rec_2',        
  'fraud_rec_3',        
  'fraud_split',        
  'fraud_testing',      
  'fraud_training',     
  'fraud_wkfl_1',       
  'fraud_wkfl_2',       
  'fraud_wkfl_3',       
  'fraud_wkfl_4',       
  'fraud_wkfl_5',       
  'fraud_wkfl_6',       
  'lc',                 
  'lc_default_fit',     
  'lc_final_fit',       
  'lc_folds',           
  'lc_grid',            
  'lc_rec',             
  'lc_split',           
  'lc_testing',         
  'lc_training',        
  'lc_xgb_tune_wkfl',   
  'lc_xgb_wkfl',        
  'lr_spec',            
  'peek_1',             
  'peek_2',             
  'peek_3',             
  'peek_4',             
  'peek_5',
  'plot_1',             
  'plot_2',             
  'plot_3',             
  'tune_fit',           
  'xgb_spec',           
  'xgb_spec_default',   
  'xgb_spec_tuning'    
  )

# Tests
expected_object_names_result <- sum(ls() %in% expected_object_names) == length(expected_object_names)

expected_cols <- read_csv('https://www.dropbox.com/scl/fi/zfzi7w4985t0tnmaua7ux/expected_cols_10.csv?rlkey=bi6n2m35jw3i9c3j56jvmfm8g&dl=1', show_col_types=F) 

if(!expected_object_names_result){
  error_content <- paste(expected_object_names[!expected_object_names %in% ls()], collapse = ',')
  stop(paste("Expected objects not found in the environment:",error_content))
}

in_mem <- lapply(mget(expected_object_names), colnames)

found_cols <- expected_cols %>% 
  left_join(in_mem[!unlist(lapply(in_mem,is.null))] %>%
              enframe() %>%
              unnest(value) %>%
              rename(tibble = name,
                     colname = value) %>% 
              mutate(was_found = 1),
            by = c("tibble" = "tibble", "colname"="colname"))

if(sum(is.na(found_cols$was_found)) == 0){
  message('All naming tests passed. But please be sure to restart your session and run your WHOLE script from beginning to end.')
} else {
  message("Uh oh. Couldn't find the column(s) below:")
  found_cols %>% 
    filter(is.na(was_found)) %>% 
    select(-was_found) %>% 
    print()
}
