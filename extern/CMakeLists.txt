include(FetchContent)

FetchContent_Declare(
  kmeans 
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG 04ca0aadd96ee5e433ef0a383006946c41a364e3
)

FetchContent_MakeAvailable(kmeans)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG ea57cbd4a0ea5574caefd6cd358f232a3813e639
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG afb49e269e02000373c55ccc982a4817be2b9d9d
)

FetchContent_MakeAvailable(aarand)
