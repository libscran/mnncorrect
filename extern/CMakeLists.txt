include(FetchContent)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/knncolle/knncolle
  GIT_TAG master # ^2.0.0
)

FetchContent_MakeAvailable(knncolle)
