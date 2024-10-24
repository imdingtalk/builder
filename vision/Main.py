import imutils
import numpy as np
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
from PyPlcnextRsc.tools import DataTypeStore
from PyPlcnextRsc import Device, RscVariant
from PyPlcnextRsc.Arp.Plc.Gds.Services import IDataAccessService, WriteItem


class ColorLabeler:
	def __init__(self):
		colors = OrderedDict({
			"red": (255, 0, 0),
			"yellow": (255, 255,0 ),
			"blue": (0, 0, 255),
			"green":(0,255,0)
		})
		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []

		for (i, (name, rgb)) in enumerate(colors.items()):
			self.lab[i] = rgb
			self.colorNames.append(name)

		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

	def label(self, image, c):
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]

		minDist = (np.inf, None)

		for (i, row) in enumerate(self.lab):
			d = dist.euclidean(row[0], mean)

			if d < minDist[0]:
				minDist = (d, i)

		return self.colorNames[minDist[1]]

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)

		if len(approx) == 3:
			shape = "triangle"

		elif len(approx) == 4:
			shape = "square"

		else:
			shape = "circle"

		return shape


def detect_shapes_and_colors(image):
	blurred = cv2.GaussianBlur(image, (5, 5), 0)
	gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
	thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	sd = ShapeDetector()
	cl = ColorLabeler()

	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	shape = sd.detect(c)
	color = cl.label(lab, c)

	if color=="red" and shape=="triangle":
		data=1
	elif color=="green" and shape=="triangle":
		data=2
	elif color=="blue" and shape=="triangle":
		data=3
	elif color == "red" and shape == "square":
		data=4
	elif color=="green" and shape=="square":
		data=5
	elif color=="blue" and shape=="square":
		data=6
	elif color == "red" and shape == "circle":
		data=7
	elif color=="green" and shape=="circle":
		data=8
	elif color=="blue" and shape=="circle":
		data=9
	elif color == "yellow" and shape == "circle":
		data=10
	elif color == "yellow" and shape == "triangle":
		data=11
	elif color == "yellow" and shape == "square":
		data=12
	else:
		data=0

	c = c.astype("int")
	text = "{} {}".format(color, shape)
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

	cv2.imshow("Image", image)
	return data

def Enum_device(tlayerType, deviceList):
	"""
	ch:枚举设备 | en:Enum device
	nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
	"""
	ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
	if ret != 0:
		print("enum devices fail! ret[0x%x]" % ret)
		sys.exit()

	if deviceList.nDeviceNum == 0:
		print("find no device!")
		sys.exit()

	print("Find %d devices!" % deviceList.nDeviceNum)

	for i in range(0, deviceList.nDeviceNum):
		mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
		if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
			print("\ngige device: [%d]" % i)
			# 输出设备名字
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
				strModeName = strModeName + chr(per)
			print("device model name: %s" % strModeName)
			# 输出设备ID
			nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
			nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
			nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
			nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
			print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
		# 输出USB接口的信息
		elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
			print("\nu3v device: [%d]" % i)
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
				if per == 0:
					break
				strModeName = strModeName + chr(per)
			print("device model name: %s" % strModeName)

			strSerialNumber = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
				if per == 0:
					break
				strSerialNumber = strSerialNumber + chr(per)
			print("user serial number: %s" % strSerialNumber)


def enable_device(nConnectionNum):
	"""
	设备使能
	:param nConnectionNum: 设备编号
	:return: 相机, 图像缓存区, 图像数据大小
	"""
	# ch:创建相机实例 | en:Creat Camera Object
	cam = MvCamera()

	# ch:选择设备并创建句柄 | en:Select device and create handle
	# cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
	stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

	ret = cam.MV_CC_CreateHandle(stDeviceList)
	if ret != 0:
		print("create handle fail! ret[0x%x]" % ret)
		sys.exit()



	# ch:打开设备 | en:Open device
	ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
	if ret != 0:
		print("open device fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
	if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
		nPacketSize = cam.MV_CC_GetOptimalPacketSize()
		if int(nPacketSize) > 0:
			ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
			if ret != 0:
				print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
		else:
			print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

	# ch:设置触发模式为off | en:Set trigger mode as off
	ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
	if ret != 0:
		print("set trigger mode fail! ret[0x%x]" % ret)
		sys.exit()


	ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
	# 从这开始，获取图片数据
	# ch:获取数据包大小 | en:Get payload size
	stParam = MVCC_INTVALUE()
	memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
	# MV_CC_GetIntValue，获取Integer属性值，handle [IN] 设备句柄
	# strKey [IN] 属性键值，如获取宽度信息则为"Width"
	# pIntValue [IN][OUT] 返回给调用者有关相机属性结构体指针
	# 得到图片尺寸，这一句很关键
	# payloadsize，为流通道上的每个图像传输的最大字节数，相机的PayloadSize的典型值是(宽x高x像素大小)，此时图像没有附加任何额外信息
	ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
	if ret != 0:
		print("get payload size fail! ret[0x%x]" % ret)
		sys.exit()

	nPayloadSize = stParam.nCurValue

	# ch:开始取流 | en:Start grab image
	ret = cam.MV_CC_StartGrabbing()
	if ret != 0:
		print("start grabbing fail! ret[0x%x]" % ret)
		sys.exit()
	#  返回获取图像缓存区。
	data_buf = (c_ubyte * nPayloadSize)()
	#  date_buf前面的转化不用，不然报错，因为转了是浮点型
	return cam, data_buf, nPayloadSize


def get_image(data_buf, nPayloadSize):
	"""
	获取图像
	:param data_buf:
	:param nPayloadSize:
	:return: 图像
	"""
	# 输出帧的信息
	stFrameInfo = MV_FRAME_OUT_INFO_EX()
	# void *memset(void *s, int ch, size_t n);
	# 函数解释:将s中当前位置后面的n个字节 (typedef unsigned int size_t )用 ch 替换并返回 s
	# memset:作用是在一段内存块中填充某个给定的值，它是对较大的结构体或数组进行清零操作的一种最快方法
	# byref(n)返回的相当于C的指针右值&n，本身没有被分配空间
	# 此处相当于将帧信息全部清空了
	memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

	# 采用超时机制获取一帧图片，SDK内部等待直到有数据时返回，成功返回0
	ret: object = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
	if ret == 0:
		print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
			stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
	else:
		print("no data[0x%x]" % ret)

	image = np.asarray(data_buf)  # 将c_ubyte_Array转化成ndarray得到（3686400，）
	image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))  # 根据自己分辨率进行转化
	return image


def close_device(cam, data_buf):
	"""
	关闭设备
	:param cam:
	:param data_buf:
	"""
	# ch:停止取流 | en:Stop grab image
	ret = cam.MV_CC_StopGrabbing()
	if ret != 0:
		print("stop grabbing fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:关闭设备 | Close device
	ret = cam.MV_CC_CloseDevice()
	if ret != 0:
		print("close deivce fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:销毁句柄 | Destroy handle
	ret = cam.MV_CC_DestroyHandle()
	if ret != 0:
		print("destroy handle fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	del data_buf


if __name__ == "__main__":
	from MvCameraControl_class import *
	# 获得设备信息
	deviceList = MV_CC_DEVICE_INFO_LIST()
	tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

	# ch: 枚举设备 | en:Enum device
	# nTLayerType[IN] 枚举传输层 ，pstDevList[OUT] 设备列表
	Enum_device(tlayerType, deviceList)

	# 获取相机和图像数据缓存区
	cam, data_buf, nPayloadSize = enable_device(0)  # 选择第一个设备

	TypeStore = DataTypeStore.fromString(
		"""
		TYPE
		udt_ImageClassify : STRUCT
			index : INT;
		END_STRUCT
		END_TYPE
		""")
	with Device('192.168.1.10', secureInfoSupplier=lambda: ("admin", "585bb8ec")) as device:
		data_access_service = IDataAccessService(device)
		im = TypeStore.NewSchemaInstance("udt_ImageClassify")


		while True:
			image = get_image(data_buf, nPayloadSize)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			im.index = detect_shapes_and_colors(image)
			data_access_service.Write((WriteItem("Arp.Plc.Eclr/imageClassify.image_result", RscVariant.of(im)),))
			print(im.index)