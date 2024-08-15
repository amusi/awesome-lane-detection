![Stars](https://img.shields.io/github/stars/voldemortX/pytorch-auto-drive)![Stars](https://img.shields.io/github/stars/voldemortX/pytorch-auto-drive)# awesome-lane-detection

This repository contains content related to 2D and 3D lane detection, as well as video lane detection.

Continual improvements are being made to this repository. If you come across any relevant papers that should be included, please don't hesitate to open an issue.

[toc]

## Activities

## 2D Lane Detection

### Segmentation-Based: *Traditional Semantic Segmentation.*

([Arxiv 1712](https://arxiv.org/abs/1712.06080)) **SCNN:** Spatial as Deep: Spatial CNN for Traffic Scene Understanding. [Xingang Pan](https://dblp.uni-trier.de/search/author?author=Xingang%20Pan) et al. [AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/12301). [Code](https://github.com/XingangPan/SCNN) ![Stars](https://img.shields.io/github/stars/XingangPan/SCNN)

([Arxiv 2008](https://arxiv.org/abs/2008.13719)) **RESA:** Recurrent Feature-Shift Aggregator for Lane Detection.[Tu Zheng](https://dblp.uni-trier.de/pid/229/4199.html) et al. [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16469). [Code](https://github.com/ZJULearning/resa)![Stars](https://img.shields.io/github/stars/ZJULearning/resa)

### Segmentation-Based: *Grid Semantic Segmentation.*

([Arxiv 2004](https://arxiv.org/abs/2004.11757)) **UFLD:** Ultra Fast Structure-Aware Deep Lane Detection. [Zequn Qin](https://dblp.uni-trier.de/pid/204/2939.html) et al. [ECCV 2020](https://doi.org/10.1007/978-3-030-58586-0_17)[Code](https://github.com/cfzd/Ultra-Fast-Lane-Detection)![Stars](https://img.shields.io/github/stars/cfzd/Ultra-Fast-Lane-Detection)

([Arxiv 2105](https://arxiv.org/abs/2105.05003)) **CondLaneNet:** a Top-to-down Lane Detection Framework Based on Conditional Convolution. [Lizhe Liu](https://dblp.uni-trier.de/pid/193/8041.html) et al. [ICCV 2021](https://doi.org/10.1109/ICCV48922.2021.00375). [Code](https://github.com/aliyun/conditional-lane-detection)![Stars](https://img.shields.io/github/stars/aliyun/conditional-lane-detection)

([Arxiv 2206](https://arxiv.org/abs/2206.07389)) **UFLDv2:** Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification. [Zequn Qin](https://dblp.uni-trier.de/pid/204/2939.html) et al. [TPAMI 2022](https://ieeexplore.ieee.org/document/9795098). [Code](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)![Stars](https://img.shields.io/github/stars/cfzd/Ultra-Fast-Lane-Detection-v2)

### Segmentation-Based: *Instance Segmentation.*

([Arxiv 1817](https://arxiv.org/abs/1807.01726)) **LaneNet:** Real-Time Lane Detection Networks for Autonomous Driving. [Ze Wang](https://dblp.uni-trier.de/search/author?author=Ze%20Wang) et al. [Code](https://paperswithcode.com/paper/lanenet-real-time-lane-detection-networks-for)![Stars](https://img.shields.io/github/stars/klintan/pytorch-lanenet)

([Arxiv 2103](https://arxiv.org/abs/2103.12040)) **LaneAF:** Robust Multi-Lane Detection with Affinity Fields. [Hala Abualsaud](https://dblp.uni-trier.de/pid/288/0904.html) et al. [LRA 2021](https://doi.org/10.1109/LRA.2021.3098066). [Code](https://github.com/sel118/LaneAF)![Stars](https://img.shields.io/github/stars/sel118/LaneAF)

([Arxiv 2105](https://arxiv.org/abs/2105.05003)) **CondLaneNet:** a Top-to-down Lane Detection Framework Based on Conditional Convolution. [Lizhe Liu](https://dblp.uni-trier.de/pid/193/8041.html) et al. [ICCV 2021](https://doi.org/10.1109/ICCV48922.2021.00375). [Code](https://github.com/aliyun/conditional-lane-detection)![Stars](https://img.shields.io/github/stars/aliyun/conditional-lane-detection)

### Detection-Based: *Keypoints Detection.*

([Arxiv 2002](https://arxiv.org/abs/2002.06604)) **PINet:** Key Points Estimation and Point Instance Segmentation Approach for Lane Detection. [TITS 2021](https://doi.org/10.1109/TITS.2021.3088488). [Code](https://paperswithcode.com/paper/key-points-estimation-and-point-instance)![Stars](https://img.shields.io/github/stars/koyeongmin/PINet)

([Arxiv 2105](https://arxiv.org/abs/2105.13680)) **FOLOLane:** Focus on Local: Detecting Lane Marker from Bottom Up via Key Point. [Zhan Qu](https://dblp.uni-trier.de/pid/37/6980.html) et al. [CVPR 2021](https://doi.org/10.1109/CVPR46437.2021.01390).

([Arxiv 2204](https://arxiv.org/abs/2204.07335)) **GANet:** A Keypoint-based Global Association Network for Lane Detection. [Jinsheng Wang](https://dblp.uni-trier.de/pid/61/3609.html) et al. [CVPR 2022](https://doi.org/10.1109/CVPR52688.2022.00145). [Code](https://github.com/Wolfwjs/GANet)![Stars](https://img.shields.io/github/stars/Wolfwjs/GANet)

### Detection-Based: *Parametric Curve Detection.*

([Arxiv 2004](https://arxiv.org/abs/2004.10924)) **PolyLaneNet:** Lane Estimation via Deep Polynomial Regression. [Lucas Tabelini](https://dblp.uni-trier.de/pid/228/1702.html?q=Lucas%20Tabelini%20Torres) et al. [ICPR 2021](https://doi.org/10.1109/ICPR48806.2021.9412265). [Code](https://github.com/lucastabelini/PolyLaneNet)![Stars](https://img.shields.io/github/stars/lucastabelini/PolyLaneNet)

([Arxiv 2011](https://arxiv.org/abs/2011.04233)) **LSTR:** End-to-End Lane Shape Prediction With Transformers. [Ruijin Liu](https://dblp.uni-trier.de/pid/254/7956.html) et al. [WACV 2021](https://doi.org/10.1109/WACV48630.2021.00374). [Code](https://github.com/liuruijin17/LSTR)![Stars](https://img.shields.io/github/stars/liuruijin17/LSTR)

([Arxiv 2203](https://arxiv.org/abs/2203.02431)) **BézierLaneNet:** Rethinking Efficient Lane Detection via Curve Modeling. [Zhengyang Feng](https://dblp.uni-trier.de/pid/263/3128.html) et al. [CVPR 2022](https://doi.org/10.1109/CVPR52688.2022.01655). [Code](https://github.com/voldemortX/pytorch-auto-drive)![Stars](https://img.shields.io/github/stars/voldemortX/pytorch-auto-drive)

([Arxiv 2301](https://arxiv.org/abs/2301.06910)) **BSNet:** Lane Detection via Draw B-spline Curves Nearby. [Haoxin Chen](https://dblp.uni-trier.de/pid/271/5537.html) et al.

### Detection-Based: Line-*Proposal&Line-Anchor Detection.*

(Arxiv xxxx) **Line-CNN:** End-to-End Traffic Line Detection With Line Proposal Unit. [Xiang Li](https://dblp.uni-trier.de/pid/40/1491-41.html) et al. [T-ITS2020](https://doi.org/10.1109/TITS.2019.2890870).

([Arxiv 2010](https://arxiv.org/abs/2010.12035)) **LaneATT:** Keep Your Eyes on the Lane: Real-Time Attention-Guided Lane Detection. [Lucas Tabelini](https://dblp.uni-trier.de/pid/228/1702.html) et al. [CVPR2021](https://doi.org/10.1109/CVPR46437.2021.00036). [Code](https://github.com/lucastabelini/LaneATT)![Stars](https://img.shields.io/github/stars/lucastabelini/LaneATT)

([Arxiv 2203](https://arxiv.org/abs/2203.10350)) **CLRNet:** Cross Layer Refinement Network for Lane Detection. [Tu Zheng](https://dblp.uni-trier.de/pid/229/4199.html) et al. [CVPR2022](https://doi.org/10.1109/CVPR52688.2022.00097). [Code](https://github.com/Turoad/CLRNet)![Stars](https://img.shields.io/github/stars/Turoad/CLRNet)

([Arxiv 2301](https://arxiv.org/abs/2301.06910)) **BSNet:** Lane Detection via Draw B-spline Curves Nearby. [Haoxin Chen](https://dblp.uni-trier.de/pid/271/5537.html) et al.

([Arxiv2305](https://arxiv.org/abs/2305.00675)) **O2Former:** End-to-End Lane Detection with One-to-Several Transformer. [Kunyang Zhou](https://dblp.uni-trier.de/pid/346/0680.html) et al. [Code](https://github.com/zkyseu/O2SFormer)![Stars](https://img.shields.io/github/stars/zkyseu/O2SFormer)

([Arxiv2308](https://arxiv.org/abs/2308.10481)) **ADNet:** Lane Shape Prediction via Anchor Decomposition. [Lingyu Xiao](https://dblp.uni-trier.de/pid/355/3184.html) et al. [ICCV2023](https://doi.org/10.1109/ICCV51070.2023.00589). [Code](https://github.com/sephirex-x/adnet)![Stars](https://img.shields.io/github/stars/sephirex-x/adnet)

## 3D Lane Detection

### Mono Deteciton?

### Mutli-View Detection?

### Mapless Driving

## Video Lane Detection

### Segmentation-Based

### Detection-Based

## Datasets

## Survey

## Articles
