<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.8">
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
      <type>std::vector&lt; size_t &gt;</type>
      <name>merge_order</name>
      <anchorfile>structmnncorrect_1_1Details.html</anchorfile>
      <anchor>af091230c4416eecedeec23bccdaa3d08</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; size_t &gt;</type>
      <name>num_pairs</name>
      <anchorfile>structmnncorrect_1_1Details.html</anchorfile>
      <anchor>a104c7a9f40bfbb834422ebf1c4884b70</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>mnncorrect::Options</name>
    <filename>structmnncorrect_1_1Options.html</filename>
    <templarg>typename Dim_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>int</type>
      <name>num_neighbors</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a77a06b63da4627528962cdb62f90ca28</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>num_mads</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a4285057efa41bf177dc941e9bdba2cef</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; knncolle::Builder&lt; knncolle::SimpleMatrix&lt; Dim_, Index_, Float_ &gt;, Float_ &gt; &gt;</type>
      <name>builder</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>ab4e93734e03009bcaec7e43b09605421</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; size_t &gt;</type>
      <name>order</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a2f2a7eb9a24cb5b0df223414b7ec7146</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>automatic_order</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>adb9f44245d56eb231b87f76412505dbf</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>robust_iterations</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>af6230f843c36e15e45007376508ef12f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>robust_trim</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a33d8e8a5102579ce703156509f09b72c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>ReferencePolicy</type>
      <name>reference_policy</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a6bdc7f31d9b5a2a9aad0862677b5c829</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>mass_cap</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>abbbbae91bfc491a2aaa9f3b2b27b8b49</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structmnncorrect_1_1Options.html</anchorfile>
      <anchor>a25c060852ae2b7402ba5b5d7e46338ba</anchor>
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
      <anchor>af638206e33439818ce2f04633fcd9bb5</anchor>
      <arglist>(size_t num_dim, const std::vector&lt; size_t &gt; &amp;num_obs, const std::vector&lt; const Float_ * &gt; &amp;batches, Float_ *output, const Options&lt; Dim_, Index_, Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Details</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a762d80298d3522cb31a5504435257100</anchor>
      <arglist>(size_t num_dim, const std::vector&lt; size_t &gt; &amp;num_obs, const Float_ *input, Float_ *output, const Options&lt; Dim_, Index_, Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Details</type>
      <name>compute</name>
      <anchorfile>namespacemnncorrect.html</anchorfile>
      <anchor>a8677b66760a4587b5e7d1db3cf252173</anchor>
      <arglist>(size_t num_dim, size_t num_obs, const Float_ *input, const Batch_ *batch, Float_ *output, const Options&lt; Dim_, Index_, Float_ &gt; &amp;options)</arglist>
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
