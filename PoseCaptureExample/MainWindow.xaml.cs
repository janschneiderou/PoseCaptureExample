using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;

namespace PoseCaptureExample
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        private VideoCapture m_Webcam;
        private Mat m_frame;
      

        #region poseEstimationInitTF


        /// <summary>
        /// Our pose estimator
        /// </summary>
        private PoseNetEstimator m_posenet;

        /// <summary>
        /// A basic flag checking if process frame is ongoing or not.
        /// </summary>
        static bool inprocessframe = false;

        /// <summary>
        /// A basic flag checking if process is ongoing or not
        /// </summary>
        static bool inprocess = false;

        int myImageHeight;
        int myImageWidth;

        #endregion

        public MainWindow()
        {
            InitializeComponent();
            initTFstuff();
        }

        #region initTF stuff

        private void initTFstuff()
        {

            myImageWidth = (int)myImage.Width;
            myImageHeight = (int)myImage.Height;
            string appBaseDir = System.AppDomain.CurrentDomain.BaseDirectory;

            m_posenet = new PoseNetEstimator(frozenModelPath: appBaseDir + "models\\posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite",
                                             numberOfThreads: 4);


        }


        private Mat GetMatFromSDImage(System.Drawing.Image image)
        {
            int stride = 0;
            Bitmap bmp = new Bitmap(image);

            System.Drawing.Rectangle rect = new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height);
            System.Drawing.Imaging.BitmapData bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadWrite, bmp.PixelFormat);

            System.Drawing.Imaging.PixelFormat pf = bmp.PixelFormat;
            if (pf == System.Drawing.Imaging.PixelFormat.Format32bppArgb)
            {
                stride = bmp.Width * 4;
            }
            else
            {
                stride = bmp.Width * 3;
            }

            Image<Bgra, byte> cvImage = new Image<Bgra, byte>(bmp.Width, bmp.Height, stride, (IntPtr)bmpData.Scan0);

            bmp.UnlockBits(bmpData);

            return cvImage.Mat;
        }

        #endregion

        #region initWebcamStuff
        private void initWebcam()
        {

            m_frame = new Mat();

            // 2- Our webcam will represent the first camera found on our device
            m_Webcam = new VideoCapture(0);   ///TODO change the number here if you are using another webcam

            // 3- When the webcam capture (grab) an image, callback on ProcessFrame method
            //m_Webcam.ImageGrabbed += M_Webcam_ImageGrabbed; // event based
            m_Webcam.ImageGrabbed += M_Webcam_ImageGrabbed; ; // event based
            m_Webcam.Start();

           
        }
        #endregion


        #region imageCaptured

        private void M_Webcam_ImageGrabbed(object sender, EventArgs e)
        {


            if (!inprocess)
            {
                // Say we start to process
                inprocess = true;

                // Reinit m_frame
                m_frame = new Mat();

                // Retrieve
                m_Webcam.Retrieve(m_frame);



                // If frame is not empty, try to process it
                if (!m_frame.IsEmpty)
                {
                    //if not already processing previous frame, process it
                    if (!inprocessframe)
                    {
                        ProcessFrame(m_frame.Clone());
                    }

                    // Display keypoints and frame in imageview
                    ShowKeypoints();
                    ShowJoints();
                    Calcfeedback();
                  

                    ShowFrame();

                }
                m_frame.Dispose();
                inprocess = false;

            }


            
        }


        private void ShowFrame()
        {
            if (!m_frame.IsEmpty)
            {

                CvInvoke.Resize(m_frame, m_frame, new System.Drawing.Size(myImageWidth, myImageHeight));
                Emgu.CV.CvInvoke.Flip(m_frame, m_frame, FlipType.Horizontal);
                var bitmap = Emgu.CV.BitmapExtension.ToBitmap(m_frame);
               
                Dispatcher.BeginInvoke(new ThreadStart(delegate { drawImage(bitmap); }));
                Emgu.CV.CvInvoke.WaitKey(1); //wait a few clock cycles



            }
        }
        private void drawImage(System.Drawing.Bitmap bitmap)
        {
            System.IO.MemoryStream ms = new System.IO.MemoryStream();
            bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
            ms.Position = 0;
            BitmapImage bi = new BitmapImage();
            bi.BeginInit();
            bi.StreamSource = ms;
            bi.EndInit();
            myImage.Source = bi;
        }



        #endregion


        #region process Pose detection
        /// <summary>
        /// A method to get keypoints from a frame.
        /// </summary>
        /// <param name="frame">A copy of <see cref="m_frame"/>. It could be resized beforehand.</param>
        public void ProcessFrame(Emgu.CV.Mat frame)
        {
            if (!inprocessframe)
            {
                inprocessframe = true;
                DateTime start = DateTime.Now;

                m_posenet.Inference(frame);

                DateTime stop = DateTime.Now;
                long elapsedTicks = stop.Ticks - start.Ticks;
                TimeSpan elapsedSpan = new TimeSpan(elapsedTicks);
                Console.WriteLine(1000 / (double)elapsedSpan.Milliseconds);

                inprocessframe = false;
            }
        }

        private void ShowKeypoints()
        {
            if (!m_frame.IsEmpty)
            {
                float count = 1;
                foreach (Keypoint kpt in m_posenet.m_keypoints) // if not empty array of points
                {
                    if (kpt != null)
                    {
                        if ((kpt.position.X != -1) & (kpt.position.Y != -1)) // if points are valids
                        {
                            Emgu.CV.CvInvoke.Circle(m_frame, kpt.position,
                                                    3, new MCvScalar(200, 255, (int)((float)255 / count), 255), 2);
                        }
                        count++;
                    }
                }
            }
        }


        private void ShowJoints()
        {
            if (!m_frame.IsEmpty)
            {
                foreach (int[] joint in m_posenet.m_keypointsJoints) // if not empty array of points
                {
                    if (m_posenet.m_keypoints[joint[0]] != null & m_posenet.m_keypoints[joint[1]] != null)
                    {
                        if ((m_posenet.m_keypoints[joint[0]].position != new System.Drawing.Point(-1, -1)) &
                            (m_posenet.m_keypoints[joint[0]].position != new System.Drawing.Point(0, 0)) &
                            (m_posenet.m_keypoints[joint[1]].position != new System.Drawing.Point(-1, -1)) &
                            (m_posenet.m_keypoints[joint[1]].position != new System.Drawing.Point(0, 0))) // if points are valids
                        {
                            Emgu.CV.CvInvoke.Line(m_frame,
                                                  m_posenet.m_keypoints[joint[0]].position,
                                                  m_posenet.m_keypoints[joint[1]].position,
                                                  new MCvScalar(200, 255, 255, 255), 2);
                        }
                    }
                }
            }
        }

        #endregion


        #region TODO Dowhatever you want to do with the pose estimation here

        private void Calcfeedback() 
        {
            ///TODO change whatever you need here in order to interpret the posture of the person
            if (m_posenet.m_keypoints[(int)BodyParts.LEFT_WRIST].position.X < m_posenet.m_keypoints[(int)BodyParts.RIGHT_WRIST].position.X)
            {

                Dispatcher.BeginInvoke(new ThreadStart(delegate
                {
                    myFeedback.Foreground = System.Windows.Media.Brushes.Red;
                    myFeedback.Content = "Reset Posture";
                }));
            }
            

        }

        #endregion

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            initWebcam();
        }
    }
}
