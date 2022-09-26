# shiny
library(shiny)
library(shinydashboard)

# tidyverse
library(tidyverse)

# # NLP and statistics
# library(NLP)
library(tm)
library(e1071)
# library(gmodels)
library(stopwords)

# wordcloud
library(wordcloud)

# import
train_termfreq <- read_csv('data/train_termfreq.csv')
load('data/train_noFS.Rdata')
load('data/train_FS.Rdata')
load('data/train_freqterms_Rlib.Rdata')
load('data/train_Rlib.Rdata')

labels <- 0:5
l.classes <- c(
  'Barely True',
  'False',
  'Half True',
  'Mostly True',
  'Not Known',
  'True', 
  'No text has been entered'
)

l.mask <- c(6, 4, 3, 1, 2, 5)

# functions
applyMultinomialNB <- function(X, classes, V, prior, log_condprob, split_func) {
  
  out_data <- NULL
  
  if(X==""){
    out_data$pred_label <- 6
    
    tmp <- rep(1, 6)
    names(tmp) <- 0:5
    out_data$score <- tmp
    
    return(out_data)
  }
  
  W <- split_func(X)
  
  # drop entries not in V
  W <- unique(W[W %in% V])
  
  
  # sum log_condprob corresponding to all the words in W
  score <- log(prior) + apply(log_condprob[W,,drop=F], FUN = sum, MARGIN = 2)
  
  
  # output data
  out_data$score <- score
  out_data$pred_label <- classes[which.max(score)]
  
  return(out_data)
}

applyNB_Rlib <- function(X, classes, train_model, freqterms) {
  
  out_data <- NULL
  
  if(X==""){
    out_data$pred_label <- 6
    
    tmp <- rep(1, 6)
    names(tmp) <- 0:5
    out_data$score <- tmp
    
    return(out_data)
  }
  
  # silence warnings
  defaultW <- getOption("warn") 
  options(warn = -1)
  
  # clean vector
  X_corpus <- Corpus(VectorSource(X))
  X_corpus_clean <- tm_map(X_corpus, tolower)
  X_corpus_clean <- tm_map(X_corpus_clean, tolower)
  X_corpus_clean <- tm_map(X_corpus_clean, removeNumbers)
  X_corpus_clean <- tm_map(X_corpus_clean, removeWords, stopwords())
  X_corpus_clean <- tm_map(X_corpus_clean, removePunctuation)
  X_corpus_clean <- tm_map(X_corpus_clean, stripWhitespace)
  
  # set back warnings
  options(warn = defaultW)
  
  # wordcount
  X_dtm <- DocumentTermMatrix(X_corpus_clean)
  X_df <- as.data.frame(as.matrix(X_dtm))
  X_df <- X_df[,freqterms[freqterms %in% colnames(X_df)]]
  
  #convert to factors
  X_df_wf <- lapply(X=X_df, FUN=function(x) factor(x, levels = c(0, 1), labels = c("No", "Yes")))
  
  # prediction score
  score <- predict(train_model, X_df_wf, "raw")
  
  out_data$score <- score
  out_data$pred_label <- classes[which.max(score)]
  
  
  return(out_data)
  
}

clean.split.text <- function(text_){
  text_ <- gsub('[[:punct:]]', '', text_) # remove punctuation
  text_ <- gsub('[[:digit:]]', '', text_) # remove digits
  text_ <- gsub('[\n\r\t]', ' ', text_)   # remove these symbols
  
  text_ <- unlist(strsplit(tolower(text_), split=' ')) # lowercase, to vector
  text_ <- text_[!text_ %in% ""]  # remove empty strings
  text_ <- text_[!text_ %in% stopwords("en")] # remove stopwords
  
  return(text_)
}

# ---------------------------- APP ------------------------------------


header <- dashboardHeader(
    title="Fake News recognition", 
    titleWidth = 250
)

sidebar <- dashboardSidebar(
  sidebarMenu(
    menuItem("App", tabName = "app", icon = icon("play")),
    menuItem("Documentation", tabName = "doc", icon = icon("book")),
    menuItem("About", tabName = "about", icon = icon("info"))
  ),
  width=250
)

body <- dashboardBody(
  
  tabItems(
    
    tabItem(tabName = "app",
      fluidRow(
        column(4,
          box(width = NULL, status = "warning",
            textAreaInput(inputId='text', label='Enter the social media post to classify', rows=3),
            radioButtons(inputId="model", label="Select the model to use:",
                      c("NB with add-one smoothing" = "NB_1",
                        "NB with feature selection" = "NB_2",
                        "NB using existing R libraries" = "NB_3")),
            actionButton(inputId = "go",
                         label = "Update")
          ),
          box(width = NULL, status = "warning", solidHeader = TRUE, plotOutput("cloud"))
        ),
        column(8,
          box(width = NULL, status = "primary", solidHeader = TRUE, title="Probability of each class",
            plotOutput("barplot", height = 400)
          ),
          box(width = NULL, status = "primary", solidHeader = TRUE, title="Predicted class",
            textOutput("prediction")
          ),
        )
      ), 
    ),
            
    
    tabItem(tabName = "doc",
      tags$h1("Overview"),
  
      tags$hr(style="border-color: black;"),
      tags$p("Fake news are defined by the New York Times as ”a made-up story with an intention to deceive”, with the intent to confuse or deceive people.
      They are everywhere in our daily life, and come especially from social media platforms and applications in the online world.
      Being able to distinguish fake contents form real news is today one of the most serious challenges facing the news industry."),

      tags$p(
        "This app was created with the aim of", 
        tags$b("classifying social media posts"), 
        "using two types of", 
        tags$i("Multinomial Naive Bayes classifiers"), 
        ". Naive Bayes classifiers are powerful algorithms that are used for text data analysis and are connected to classification tasks of text in multiple classes. 
        More information about them (e.g. how the algorithm works) can be found in the first source in the references."
      ),
      
      tags$h2("Training procedure"),
      tags$p(
        "The models were trained using the dataset available on Kaggle:", 
        tags$a(href="https://www.kaggle.com/datasets/anmolkumar/fake-news-content-detection", "dataset"), 
        ". In particular, the file", 
        tags$i("train.csv"), 
        "consists of a training set with 10,240 labeled instances."  
      ),
      tags$p(
        "First of all, to evaluate the accuracy of each model, the dataset was divided into training and testing according to an 80-20% division. 
        The entire dataset was then used to train the models imported into the app." 
      ),
      
      tags$h2("Labels description"),
      tags$p(
        "The labels available for classifying the input text are:", 
        tags$ul(
          tags$li("True"), 
          tags$li("Mostly True"), 
          tags$li("Half True"), 
          tags$li("Barely True"), 
          tags$li("False"), 
          tags$li("Not Known") 
        )
      ),
      
      tags$hr(), 
      tags$h1("Trained models available"),
      tags$hr(style="border-color: black;"),
      tags$p(
        "When an input social media post is entered in the app, it is possible to classify it according to 3 different models:",
        tags$ul(
          tags$li("NB with add-one smoothing"), 
          tags$li("NB with feature selection"), 
          tags$li("NB using existing R libraries")
        ), 
        "They are all Naive Bayes classifiers and they all implement add-one smoothing. While the first two are implemented by scratch, 
        using or not the concept of feature selection, the last model has been trained using the function", 
        tags$code("naiveBayes()"), 
        "from the library", 
        tags$code("e1071")
      ),
      
      tags$hr(), 
      tags$h1("Sample texts"),
      tags$hr(style="border-color: black;"),
      tags$p(
        "We report here some examples of social media posts present in the Kaggle", 
        tags$i("test.csv"), 
        "file, which can be used to test the application and play with it.",
        box(width = NULL, solidHeader = TRUE, 
          tags$table(
            tags$tr( 
              tags$th("Sample Text")
            ), 
            tags$tr(
              tags$td("Obamacare has caused millions of full-time jobs to become part-time.")
            ), 
            tags$tr(
              tags$td("You cant read a speech by George Washington . . . without hearing him reference God, the Almighty.")
            ), 
            tags$tr(
              tags$td("Florida ranks last in the ratio of employees to residents...And Florida is dead last in the nation in state employee payroll expenditures per resident.")
            ), 
            tags$tr(
              tags$td("The House of Delegates budget bill cuts $50 million from education.") 
            ), 
            tags$tr(
              tags$td("At least Obama didn't marry his cousin, as Giuliani did.") 
            ), 
            tags$tr(
              tags$td("Says his reform efforts improved performance at all 10 low-performing schools in Palm Beach, Florida.") 
            ), 
            tags$tr(
              tags$td("As the usage [of synthetic marijuana] has dramatically increased, instances of violence, bodily harm and even death have risen with it.") 
            ), 
            tags$tr(
              tags$td("Says David Dewhurst explicitly advocated a guest worker program for all illegal immigrants.") 
            ), 
            tags$tr(
              tags$td("A new Colorado law literally allows residents to print ballots from their home computers, then encourages them to turn ballots over to collectors.") 
            ), 
            tags$tr(
              tags$td("Says every school will be negatively impacted if Education Stability Fund is not tapped.") 
            )
          )
        )
      ),
      
      tags$hr(), 
      tags$h1("References"),
      tags$hr(style="border-color: black;"),
      tags$ul(
        tags$li("C. D. Manning, Chapter 13, Text Classification and Naive Bayes, in Introduction to Information Retrieval, Cambridge University Press, 2008."), 
        tags$li("Fake News Content Detection, KAGGLE data set:", 
                tags$a(href="https://www.kaggle.com/datasets/anmolkumar/fake-news-content-detection", "kaggle/fake-news-content-detection")) 
      )
    ), 
    
    tabItem(tabName = "about",
      fluidRow(
        column(6, 
          box(width = NULL, solidHeader = TRUE, 
            tags$h1("About the Project"),
            tags$hr(style="border-color: black;"),
            tags$p(
              "This app was created as a final project of the Advanced Statistics for Physics Analysis course at the University of Padua. 
              More information about the course can be found here:", 
              tags$a(href="https://en.didattica.unipd.it/off/2021/LM/SC/SC2443/000ZZ/SCP8082557/N0", "AS4PA"), 
              "."
            ),
            tags$p( 
              "The source code for the training of the models and the evaluation of their accuracy is available on GitHub:", 
              tags$a(href="https://github.com/PaoloZinesi/NaiveBayesClassifier_Fake_News_Recognition/blob/main/NB_fake_news.ipynb", "notebook"), 
              "."
            ), 
            tags$p( 
              "The source code for the Shiny app is also available on GitHub:",
              tags$a(href="https://github.com/PaoloZinesi/NaiveBayesClassifier_Fake_News_Recognition/tree/main/app", "app"), 
              "."
            )
          )
        ), 
        column(6, 
          box(width = NULL, solidHeader = TRUE, 
            tags$h1("About the Authors"),
            tags$hr(style="border-color: black;"),
            tags$p(
              "Hi! We are Master's student in Physics of Data at the University of Padua, a Master’s degree that provides new theoretical and computational 
              tools to tackle the explosion of datasets within the physicist mindset."
            ),
            tags$p(
              "You can find our work, as well as contact information, in our GitHub repositories:",
              tags$ul(
                tags$li(tags$a(href="https://github.com/PaoloZinesi", "Paolo Zinesi")), 
                tags$li(tags$a(href="https://github.com/NicolaZomer", "Nicola Zomer"))
              )
            )
          )     
        )
      )
    )
  )
)

  
ui <- dashboardPage(
    header,
    sidebar,
    body
)


server <- function(input, output) {
  # input
  data <- eventReactive(input$go, {
    input$text
  })
  
  # output
  output$cloud <- renderPlot({
    par(mar = rep(0, 4))
    wordcloud(
      words = train_termfreq$words,
      freq = train_termfreq$freq,
      max.words=250,
      random.order=FALSE
    )
  }) 
  
  fn_output <-  eventReactive(input$go, {
    switch(input$model, 
           "NB_1" = applyMultinomialNB(data(), labels, fn_training$V, fn_training$prior, fn_training$log_condprob, clean.split.text),
           "NB_2" = applyMultinomialNB(data(), labels, fn_training_features$V, fn_training_features$prior, fn_training_features$log_condprob, clean.split.text),
           "NB_3" = applyNB_Rlib(data(), labels, data_classifier_wf, train_freqterms_Rlib)
    )
  })
  output$prediction <- renderText({
    l.classes[fn_output()$pred_label+1]
  })
  output$barplot <- renderPlot({
    ggplot()+
      geom_bar(aes(x=factor(l.classes[l.mask], levels=l.classes[l.mask]), y=exp(fn_output()$score[l.mask])/sum(exp(fn_output()$score[l.mask]))), stat='identity', fill='forestgreen') +
      labs(
        x='Classes', 
        y='Normalized Score'
      )+
      theme_bw()+
      theme(axis.text=element_text(size=12),
           axis.title=element_text(size=14,face="bold"))
  })
  
}

shinyApp(ui = ui, server = server)
