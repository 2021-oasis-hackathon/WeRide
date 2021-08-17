import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box_PIL, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

######## distance
import math
import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog

from settings import CALIB_FILE_NAME, PERSPECTIVE_FILE_NAME, UNWARPED_SIZE, ORIGINAL_SIZE
########
from os import system

class DigitalFilter:

    def __init__(self, vector, b, a):
        self.len = len(vector)
        self.b = b.reshape(-1, 1)
        self.a = a.reshape(-1, 1)
        self.input_history = np.tile(vector.astype(np.float64), (len(self.b), 1))
        self.output_history = np.tile(vector.astype(np.float64), (len(self.a), 1))
        self.old_output = np.copy(self.output_history[0])

    def output(self):
        return self.output_history[0]

    def speed(self):
        return self.output_history[0] - self.output_history[1]

    def new_point(self, vector):
        self.input_history = np.roll(self.input_history, 1, axis=0)
        self.old_output = np.copy(self.output_history[0])
        self.output_history = np.roll(self.output_history, 1, axis=0)
        self.input_history[0] = vector
        self.output_history[0] = (np.matmul(self.b.T, self.input_history) - np.matmul(self.a[1:].T, self.output_history[1:]))/self.a[0]
        return self.output()

    def skip_one(self):
        self.new_point(self.output())


def area(bbox):
    return float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))


class Car:
    def __init__(self, bounding_box, first=False, warped_size=None, transform_matrix=None, pix_per_meter=None, object=None):
        self.warped_size = warped_size
        self.transform_matrix = transform_matrix
        self.pix_per_meter = pix_per_meter
        self.has_position = self.warped_size is not None \
                            and self.transform_matrix is not None \
                            and self.pix_per_meter is not None

        self.filtered_bbox = DigitalFilter(bounding_box, 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.position = DigitalFilter(self.calculate_position(bounding_box, object), 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.found = True
        self.num_lost = 0
        self.num_found = 0
        self.display = first
        self.fps = 20

    def calculate_position(self, bbox, object):
        if (self.has_position) and (object == 0):   #vehicle
            pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
            dst = cv2.perspectiveTransform(pos, self.transform_matrix).reshape(-1, 1)
            return np.array(((self.warped_size[1]-dst[1])/self.pix_per_meter[1]+0.9)*1.3*1.3*1.3*1.3) ##초점 조절
        elif (self.has_position) and (object == 1):   #traffic_light
            pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
            dst = cv2.perspectiveTransform(pos, self.transform_matrix).reshape(-1, 1)
            return np.array((self.warped_size[1]-dst[1])/(self.pix_per_meter[1]*1.3*1.3*1.3*1.3)) ## 초점 조절
        elif (self.has_position) and (object == 2):    #pedestrian
            pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
            dst = cv2.perspectiveTransform(pos, self.transform_matrix).reshape(-1, 1)
            return np.array((self.warped_size[1]-dst[1])/(self.pix_per_meter[1]*1.3*1.3*1.3*1.3)) ## 초점 조절
        else:
            return np.array([0])

    def draw(self, img, color=(255, 0, 0), thickness=2):
        if self.display:
            window = self.filtered_bbox.output().astype(np.int32)
            # cv2.rectangle(img, (window[0], window[1]), (window[2], window[3]), color, thickness)
            if self.has_position and self.position.output()[0] :
                cv2.putText(img, "{:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[3]+20)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))
                cv2.putText(img, "{:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[3]+20)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))
        return self.position.output()[0]

    def draw_speed(self, img, color=(255, 0, 0), thickness=2, frame_history= 0 ):
        if self.display:
            window = self.filtered_bbox.output().astype(np.int32)
            if self.has_position and self.position.output()[0] : # 1프레임 당 이동거리(m)*FPS*3600/1000=1초 당 이동거리(m)*3600/1000=1시간 당 이동거리(m)/1000=1시간 당 이동거리(km)
                cv2.putText(img, "RS: {:6.2f}km/h".format((self.position.output()[0]-frame_history)*self.fps*3.6), (int(window[0]), int(window[3]+35)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))
                cv2.putText(img, "RS: {:6.2f}km/h".format((self.position.output()[0]-frame_history)*self.fps*3.6), (int(window[0]), int(window[3]+35)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))
            # print("현재거리:",self.position.output()[0])
            # print("이전 거리: ", frame_history)
        return (self.position.output()[0]-frame_history)*self.fps*3.6
############

def user_feedback(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # 사용자 피드백
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

#시나리오별 점수 계산
def caculate_drive(xyxy, im0, label, c, perspective_data,
                  vehicle_frame_history,
                  vehicle_speed_history, 
                  traffic_light_frame_history,
                  pedestrian_frame_history): #bbox/img/label/class/matrix/frame_history for 상대속도 
    ####### 변환 매트릭스 수정 부분
    transform_matrix = perspective_data["perspective_transform"]
    transfrom_matrix_reverse = transform_matrix[::-1] ##신호등 거리 계산을 위해 변환행렬을 위아래 대칭으로 바꿈
    pixels_per_meter = perspective_data['pixels_per_meter']
    orig_points = perspective_data["orig_points"]
    warped_size = (364, 640)
    warped_size_light = (364, 640)
    distance=0
    relative_speed=0
    score_case=[] #case 0=Null/1=차량거리 유지/2=차량 급감속/3=신호위반/4=보행자 발견시 서행/5=차선이탈/6=정지선 지킴

    mid_point=int(im0.shape[1]/2)
    bbox = np.array((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])))
    # ####수정
    feedback_xyxy=[0,50,1000,10]
    if label == 'manholecover' or label == 'pothole' or label == 'roadcrack': # 단순 사용자 피드백
        plot_one_box(xyxy, im0, label=label, color=(0,0,255), line_thickness=opt.line_thickness)
        user_feedback(feedback_xyxy, im0, label='Watch out for obstacle!', color=(0,0,255), line_thickness=5)

    elif label.split('_')[0] == 'Vehicle' : #거리 추가/ 거리 및 상대 속도에 따른 충돌 위험 피드백
        plot_one_box(xyxy, im0, label=label, color=(128,128,128), line_thickness=opt.line_thickness)
        
        if bbox[0] <= mid_point and mid_point <= bbox[2]:
            car = Car(bbox, True, warped_size, transform_matrix, pixels_per_meter, object= 0) #차량 거리 계산
            relative_speed=car.draw_speed(im0, color=(0, 0, 255), thickness=2, frame_history=vehicle_frame_history)
            distance=car.draw(im0, color=(0, 0, 255), thickness=2)
            if distance < 1.5 : #거리가 가까우면 충돌 위험 피드백
                user_feedback(feedback_xyxy, im0, label='Too close!', color=(0,0,200), line_thickness=5)
                if not (vehicle_frame_history < 1.5) :  #이전 프레임에서 충돌 위험이 없었을 때만 score_case에 추가(중복 방지) 
                    score_case.append(1)
            if relative_speed < -5:
                if not (vehicle_speed_history < -5):     #이전 프레임에서 급감속이 없었을 때만 score_case에 추가(중복 방지)
                    score_case.append(2)
            vehicle_frame_history=distance
            vehicle_speed_history=relative_speed
        else:
            car = Car(bbox, True, warped_size, transform_matrix, pixels_per_meter, object= 0) #차량 거리 계산
            car.draw(im0, color=(0, 0, 255), thickness=2)

    elif label.split('_')[0] == 'TrafficLight' : #거리 추가/ 거리에 따른 정차 유무 판단 후 피드백
        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
        # if label == 'TrafficLight_Red':
        if bbox[0] <= mid_point and mid_point <= bbox[2]:
            car2 = Car(bbox, True, warped_size_light, transfrom_matrix_reverse, pixels_per_meter, object= 1) #신호등 거리 계산
            relative_speed=car2.draw_speed(im0, color=(0, 0, 255), thickness=2, frame_history=traffic_light_frame_history)
            traffic_light_frame_history=car2.draw(im0, color=(0, 0, 255), thickness=2)
        else:
            car2 = Car(bbox, True, warped_size_light, transfrom_matrix_reverse, pixels_per_meter, object= 1)
            car2.draw(im0, color=(0, 0, 255), thickness=2)
        
    elif label.split('_')[0] == 'Pedestrian' : #거리 추가/ 거리에 따른 피드백
        plot_one_box(xyxy, im0, label=label, color=(0,0,200), line_thickness=opt.line_thickness)

        if bbox[0] <= mid_point and mid_point <= bbox[2]:
            car3 = Car(bbox, True, warped_size, transform_matrix, pixels_per_meter, object= 2) 
            relative_speed=car3.draw_speed(im0, color=(0, 0, 255), thickness=2, frame_history=pedestrian_frame_history)
            pedestrian_frame_history=car3.draw(im0, color=(0, 0, 255), thickness=2)
        
    elif label == 'RoadMark_StopLine' or label == 'RoadMark_Crosswalk': # 거리 추가/ 빨간불일때,상대속도로 급정차 유무 판단 및 정지선 지킴 유무 판단 
        plot_one_box(xyxy, im0, label=label, color=(0,0,150), line_thickness=opt.line_thickness)

    ####
    # 차선 이탈유무 판단 추가
    ####
    else: #traffic sign 피드백, RoadMark 피드백 / 현재 사용 x
        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)

    return vehicle_frame_history,vehicle_speed_history, traffic_light_frame_history, pedestrian_frame_history, score_case

def detect(opt):
    score_result=[]
    ################ distance part
    PERSPECTIVE_FILE_NAME = 'projection.p'
    with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
        perspective_data = pickle.load(f)
    #### 이전 프레임 거리
    vehicle_frame_history=0
    vehicle_speed_history=0
    traffic_light_frame_history=0
    pedestrian_frame_history=0
    ###################

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    result=[frame,dataset.frames,0] #프레임/점수를 반환
                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        vehicle_frame_history,vehicle_speed_history, traffic_light_frame_history,pedestrian_frame_history, result[2] = caculate_drive(
                                                                                                                                        xyxy,im0,label,c,perspective_data,
                                                                                                                                        vehicle_frame_history,
                                                                                                                                        vehicle_speed_history, 
                                                                                                                                        traffic_light_frame_history,
                                                                                                                                        pedestrian_frame_history)
                        if len(result[2])!=0:
                            score_result.append(result)
                            print("score_result :",score_result)
                            print("result :",result)

                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                            
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]                        
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))                        
                    vid_writer.write(im0)
                    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print("score_case:", score_result )
    print("fps:",fps)

    return score_result, fps

def result_info(score_result, fps): #case 0=Null/1=차량거리 유지/2=차량 급감속/3=신호위반/4=보행자 발견시 서행/5=차선이탈/6=정지선 지킴
    total_time=score_result[0][1]/fps
    score=100
    s=f""
    for time, times, cases in score_result:
        for case in cases:
            if case == 0:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: Null')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: Null'
                score-=1
            elif case == 1:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: 차간 거리 유지 감점')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: 차간 거리 유지 감점'
                score-=1
            elif case == 2:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: 급감속 감점')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: 급감속 감점'
                score-=1
            elif case == 3:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: 신호위반 감점')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: 신호위반 감점'
                score-=1
            elif case == 4:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: 보행자 발견시 서행 감점')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: 보행자 발견시 서행 감점'
                score-=1
            elif case == 5:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: 차선 이탈 감점')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: 차선 이탈 감점'
                score-=1
            elif case == 6:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: 정지선 지킴 감점')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: 정지선 지킴 감점'
                score-=1
            else:
                print(f'00:{time/fps:.0f}/00:{total_time:.0f}: Null')
                s+=f'\n00:{time/fps:.0f}/00:{total_time:.0f}: Null'
                score-=1

    print("총점:",score)

    return s,score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best(epoch 16 final).pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../test_video/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='../output/', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            score_result, fps = detect(opt=opt)
    
    score_table, total_score = result_info(score_result,fps)

    system("python detect_lane.py")