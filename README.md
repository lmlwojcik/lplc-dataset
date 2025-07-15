# LPRD Dataset - License Plate Readability Detection

This is the repository for the LPRD Dataset, presented in the paper "A New Dataset for Readability Classification for License Plate Recognition". This dataset is comprised of 10,200 images, with 12,687 license plates annotated in total. Each license plate is annotated according to readability (4 levels), OCR (for 3 of the 4 readability levels), bounding box (four points (x,y)) and plate and vehicle-wise occlusion (valid vs. occluded attributes). Dataset statistics are available below.


<table border="1px solid black">
  <tr>
    <th colspan="2">
        LPs by readability
    </th>
    <th colspan="3">
        Other attributes
    </th>
  </tr>
  <tr>
    <th>
      Class
    </th>
    <th>
      Amount
    </th>
    <th>
      Class
    </th>
    <th>
      True
    </th>
    <th>
      False
    </td>
  </tr>
  <tr>
    <td>
      Perfect
    </td>
    <td text-align='center'>
      5,535
    </td>
    <td>
      Plate characters occluded
    </td>
    <td>
      12,586
    </td>
    <td>
      101
    </td>
  </tr>
  <tr>
    <td>
      Good
    </td>
    <td>
      3,426
    </td>
    <td>
      Valid (non-occluded) vehicle
    </td>
    <td>
      12,389
    </td>
    <td>
      328
    </td>
  </tr>
  <tr>
    <td>
      Poor
    </td>
    <td>
      2,122
    </td>
    <td>
      Has OCR
    </td>
    <td>
      11,083
    </td>
    <td>
      1604
    </td>
  </tr>
  <tr>
    <td>
      Illegible
    </td>
    <td>
      1,604
    </td>
    <td>
      Total LPs
    </td>
    <td colspan=2>
      12,687
    </td>
  </tr>
</table>

The LPRD dataset is available under request. If you are interested, please contact us (lmlwojcik@inf.ufpr.br or menotti@inf.ufpr.br) in an e-mail titled "2025 LPRD Request Form". Please inform your name, affiliation and purpose of use. Also inform one or two of your recent publications (up to 5 years), if any.

All samples in the dataset can only be used by the applicant and only used for academic research. It may not be used for commercial usage, and use in publications must be properly acknowledged. The BibTeX citation is available below.

```
Hello world
```