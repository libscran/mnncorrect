<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>compute.hpp</name>
    <path>mnncorrect/</path>
    <filename>compute_8hpp.html</filename>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <class kind="struct">mnncorrect::Details</class>
    <namespace>mnncorrect</namespace>
  </compound>
  <compound kind="file">
    <name>mnncorrect.hpp</name>
    <path>mnncorrect/</path>
    <filename>mnncorrect_8hpp.html</filename>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <includes id="ReferencePolicy_8hpp" name="ReferencePolicy.hpp" local="yes" import="no" module="no" objc="no">ReferencePolicy.hpp</includes>
    <includes id="compute_8hpp" name="compute.hpp" local="yes" import="no" module="no" objc="no">compute.hpp</includes>
    <namespace>mnncorrect</namespace>
  </compound>
  <compound kind="file">
    <name>Options.hpp</name>
    <path>mnncorrect/</path>
    <filename>Options_8hpp.html</filename>
    <includes id="ReferencePolicy_8hpp" name="ReferencePolicy.hpp" local="yes" import="no" module="no" objc="no">ReferencePolicy.hpp</includes>
    <class kind="struct">mnncorrect::Options</class>
    <namespace>mnncorrect</namespace>
  </compound>
  <compound kind="file">
    <name>parallelize.hpp</name>
    <path>mnncorrect/</path>
    <filename>parallelize_8hpp.html</filename>
    <namespace>mnncorrect</namespace>
  </compound>
  <compound kind="file">
    <name>ReferencePolicy.hpp</name>
    <path>mnncorrect/</path>
    <filename>ReferencePolicy_8hpp.html</filename>
    <namespace>mnncorrect</namespace>
  </compound>
  <compound kind="struct">
    <name>mnncorrect::Details</name>
    <filename>structmnncorrect_1_1Details.html</filename>
    <member kind="variable">
      <type>std::vector&lt; std::size_t &gt;</type>
      <name>merge_order</name>
      <anchorfile>structmnncorrect_1_1Details.html</anchorfile>
      <anchor>a08d8afbc5fc5d398dee8181dbd045dc9</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; unsigned long long &gt;</type>
      <name>num_pairs</name>
      <anchorfile>structmnncorrect_1_1Details.html</anchorfile>
      <anchor>ab0489fb429ee3efe9dedf421b3e8ae46</anchor>
      <arglist></arglist>
    </member>
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
      <type>double</type>
      <name>num_mads</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>aa1257dbad59eb1fe7f3f74743b8a2aa7</anchor>
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
      <type>std::vector&lt; std::size_t &gt;</type>
      <name>order</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>ab2fb06491f25df8314b67fe8ec8bc3fa</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>automatic_order</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>afb71da89c11ab9a91b2542460fe0b652</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>robust_iterations</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a4a52b39e6d30a5f97358b0a5478c765d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>robust_trim</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a5383753859038cf0a1e7bc8f7b5ae025</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>ReferencePolicy</type>
      <name>reference_policy</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>aacb2b4821b13989b2c904cacceab009b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Index_</type>
      <name>mass_cap</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a24eeb8491341845c33b8fdf59bc081f5</anchor>
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
    <class kind="struct">mnncorrect::Details</class>
    <class kind="struct">mnncorrect::Options</class>
    <member kind="enumeration">
      <type></type>
      <name>ReferencePolicy</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a628f306b43e4b2a96382aa7025940694</anchor>
      <arglist></arglist>
      <enumvalue file="namespacemnncorrect.html" anchor="a628f306b43e4b2a96382aa7025940694aa84cc046d48610b05c21fd3670d0c829">INPUT</enumvalue>
      <enumvalue file="namespacemnncorrect.html" anchor="a628f306b43e4b2a96382aa7025940694aa4932cf9fffdd4b9b00f4d289ba7f205">MAX_SIZE</enumvalue>
      <enumvalue file="namespacemnncorrect.html" anchor="a628f306b43e4b2a96382aa7025940694a7a297b1e360b1ac7fcd220b55691b3a0">MAX_VARIANCE</enumvalue>
      <enumvalue file="namespacemnncorrect.html" anchor="a628f306b43e4b2a96382aa7025940694a2206c5fadc2d0a3af1186b871642cfdf">MAX_RSS</enumvalue>
    </member>
    <member kind="function">
      <type>Details</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>adb405eb08977f1680f1a717db3afd4a8</anchor>
      <arglist>(std::size_t num_dim, const std::vector&lt; Index_ &gt; &amp;num_obs, const std::vector&lt; const Float_ * &gt; &amp;batches, Float_ *output, const Options&lt; Index_, Float_, Matrix_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Details</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>ad03058d929f26376d9a06ccd8317c382</anchor>
      <arglist>(std::size_t num_dim, const std::vector&lt; Index_ &gt; &amp;num_obs, const Float_ *input, Float_ *output, const Options&lt; Index_, Float_, Matrix_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Details</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a3d0ccdc1c91779a0aecac6ad73fe15ab</anchor>
      <arglist>(std::size_t num_dim, Index_ num_obs, const Float_ *input, const Batch_ *batch, Float_ *output, const Options&lt; Index_, Float_, Matrix_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>parallelize</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a1b556e96c8995a0dba23e412b75c838a</anchor>
      <arglist>(int num_workers, Task_ num_tasks, Run_ run_task_range)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>C++ library for MNN correction</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
