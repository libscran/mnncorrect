<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>mnncorrect.hpp</name>
    <path>mnncorrect/</path>
    <filename>mnncorrect_8hpp.html</filename>
    <includes id="utils_8hpp" name="utils.hpp" local="yes" import="no" module="no" objc="no">utils.hpp</includes>
    <class kind="struct">mnncorrect::Options</class>
    <namespace>mnncorrect</namespace>
  </compound>
  <compound kind="file">
    <name>utils.hpp</name>
    <path>mnncorrect/</path>
    <filename>utils_8hpp.html</filename>
    <namespace>mnncorrect</namespace>
  </compound>
  <compound kind="struct">
    <name>mnncorrect::Options</name>
    <filename>structmnncorrect_1_1Options.html</filename>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <templarg>class Matrix_</templarg>
    <member kind="variable">
      <type>int</type>
      <name>num_neighbors</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>ab5023728bd400bacaa74812cb8a56b5d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_steps</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a2ec5fc17723c0c92611c942a77b1f83f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; knncolle::Builder&lt; Index_, Float_, Float_, Matrix_ &gt; &gt;</type>
      <name>builder</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a7143d3f30635f258730d15fc2eff8e40</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>MergePolicy</type>
      <name>merge_policy</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>af9da9756546454fdd88a9ab7afc7a34e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>aa1e850cce8e765cf5e7b03ea3d240ce2</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>mnncorrect</name>
    <filename>namespacemnncorrect.html</filename>
    <class kind="struct">mnncorrect::Options</class>
    <member kind="typedef">
      <type>std::size_t</type>
      <name>BatchIndex</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a67ab090aec935cd48ad2e5d3383926f7</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>MergePolicy</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a5a9c9f9569fcad76a5e735cd8fae0197</anchor>
      <arglist></arglist>
      <enumvalue file="namespacemnncorrect.html" anchor="a5a9c9f9569fcad76a5e735cd8fae0197aa84cc046d48610b05c21fd3670d0c829">INPUT</enumvalue>
      <enumvalue file="namespacemnncorrect.html" anchor="a5a9c9f9569fcad76a5e735cd8fae0197a62e5cef85d46f1a5a2144d9fd463b79e">SIZE</enumvalue>
      <enumvalue file="namespacemnncorrect.html" anchor="a5a9c9f9569fcad76a5e735cd8fae0197ace18bb9a2b22515d0cd36bca6b998bde">VARIANCE</enumvalue>
      <enumvalue file="namespacemnncorrect.html" anchor="a5a9c9f9569fcad76a5e735cd8fae0197abf1981220040a8ac147698c85d55334f">RSS</enumvalue>
    </member>
    <member kind="function">
      <type>void</type>
      <name>parallelize</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a1b556e96c8995a0dba23e412b75c838a</anchor>
      <arglist>(int num_workers, Task_ num_tasks, Run_ run_task_range)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a0ce88c6787f0f2ec73101fc234589ee6</anchor>
      <arglist>(std::size_t num_dim, const std::vector&lt; Index_ &gt; &amp;num_obs, const std::vector&lt; const Float_ * &gt; &amp;batches, Float_ *output, const Options&lt; Index_, Float_, Matrix_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>aaa1ac3e8b237d6402465f21cb5b62f20</anchor>
      <arglist>(std::size_t num_dim, const std::vector&lt; Index_ &gt; &amp;num_obs, const Float_ *input, Float_ *output, const Options&lt; Index_, Float_, Matrix_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>aee9c4838402953d68aec1596983d01ee</anchor>
      <arglist>(std::size_t num_dim, Index_ num_obs, const Float_ *input, const Batch_ *batch, Float_ *output, const Options&lt; Index_, Float_, Matrix_ &gt; &amp;options)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>C++ library for MNN correction</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
