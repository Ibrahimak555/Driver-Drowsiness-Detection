## Driver Drowsiness Detection
Drowsy driver is alerted when he/she is not attentive. The system uses features of non-alertness such as 'Eye closure', 'Yawning', and 'Head pose'.

## Demo
![DDD Demo - Eye and Yawn](https://github.com/Ibrahimak555/Driver-Drowsiness-Detection/assets/78961240/aaa5aefc-a383-4be4-98f7-dbe1ab073244)
![DDD Demo - Head Pose](https://github.com/Ibrahimak555/Driver-Drowsiness-Detection/assets/78961240/7470d794-3240-41bb-9232-a579093b47e4)

- We use the MediaPipe library which imposes a face mesh to identify points of interest in DDD from the mesh.
<table>
<tbody>
 <tr>
   <td><a><img src="Driver_Attention_System/images/MediaPipe_Landmarks.jpg" title="Face Mesh Landmarks" width="400" height="400"/></a></td>
   <td><a><img src="Driver_Attention_System/images/MediaPipe_Face_Mesh.png" title="Face Mesh imposed on face" width="400" height="400"/></a></td>
 </tr>
</tbody>
</table>

- The Points of interest for driver drowsiness are the eye (green) and mouth (black) points for 'Eye closure detection' and 'Yawn detection' respectively.
- For Head pose estimation we need the following points labelled with red colour.

<table>
<tbody>
 <tr>
   <td><a><img src="Driver_Attention_System/images/DDD_Points_of_interest.jpg" title="Points of Interest for DDD" width="500" height="500"/></a></td>
   <td><a><img src="Driver_Attention_System/images/Attentive.png" title="Attentive person" width="600" height="500"/></a></td>
 </tr>
</tbody>
</table>

---
- Using EAR (Eye Aspect Ratio), the ratio of the height of the eye to the width. If the ratio is less than 
   the threshold of 0.25 for more than 2.5 seconds, the system alerts using an alarm.
- Using the Mouth ratio, if it is more than the threshold of 0.3 for more than 4 seconds, the system alerts the driver.
- Using Solve PnP to find the point where the driver is looking and find if he/she is distracted using the head pose estimation.

<table>
<tbody>
 <tr>
   <td><a><img src="Driver_Attention_System/images/Looking_Right.png" title="Non-Attentive" width="600" height="500"/></a></td>
   <td><a><img src="Driver_Attention_System/images/Yawn.png" title="Yawn Detection" width="600" height="500"/></a></td>
 </tr> 
</tbody>
</table>

We get a system where non-attention in a person is detected and is alerted using an alarm.
