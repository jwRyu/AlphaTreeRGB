#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#define DEBUG 0
#define max(a,b) (a)>(b)?(a):(b)
#define min(a,b) (a)>(b)?(b):(a)

#define DELAYED_ANODE_ALLOC		1
#define HQUEUE_COST_AMORTIZE	1

#define NULL_LEVELROOT		0xffffffff
#define ANODE_CANDIDATE		0xfffffffe

#define dimg_idx_v(pidx) ((pidx)<<1)
#define dimg_idx_h(pidx) ((pidx)<<1)+1

#define LEFT_AVAIL(pidx,width)			(((pidx) % (width)) != 0)
#define RIGHT_AVAIL(pidx,width)			(((pidx) % (width)) != ((width) - 1))
#define UP_AVAIL(pidx,width)				((pidx) > ((width) - 1))
#define DOWN_AVAIL(pidx,width,imgsz)		((pidx) < (imgsz) - (width))

#define HQUEUE_ELEMENT_SIZE		4
#define LEVELROOT_ELEMENT_SIZE	8
#define DHIST_ELEMENT_SIZE		4
#define DIMG_ELEMENT_SIZE		1
#define ISVISITED_ELEMENT_SIZE	1

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned long uint32;
typedef unsigned long long uint64;

typedef uint8 pixel; //designed for 8-bit images

typedef struct HQueue
{
	uint32 *queue, *bottom, *cur;
	uint64 qsize;
	uint32 min_level, max_level;
}HQueue;


HQueue* hqueue_new(uint64 qsize, uint32 *dhist, uint32 dhistsize)
{
	uint32 i;
	HQueue* hqueue = (HQueue*)malloc(sizeof(HQueue));
	hqueue->queue = (uint32*)malloc((size_t)qsize * sizeof(uint32));
	hqueue->bottom = (uint32*)malloc((size_t)(dhistsize + 1) * sizeof(uint32));
	hqueue->cur = (uint32*)malloc((size_t)(dhistsize + 1) * sizeof(uint32));

	hqueue->qsize = qsize;
	hqueue->min_level = hqueue->max_level = dhistsize;

	
	int sum_hist = 0;
	for (i = 0; i < dhistsize; i++)
	{
		hqueue->bottom[i] = hqueue->cur[i] = sum_hist;
		sum_hist += dhist[i];
	}
	hqueue->bottom[dhistsize] = 0;
	hqueue->cur[dhistsize] = 1;

	return hqueue;
}

void hqueue_free(HQueue* hqueue)
{
	free(hqueue->queue);
	free(hqueue->bottom);
	free(hqueue->cur);
	free(hqueue);
}

inline void hqueue_push(HQueue* hqueue, uint32 newidx, uint32 level)
{
	hqueue->min_level = min(level, hqueue->min_level);
#if DEBUG
	assert(level < hqueue->max_level);
	assert(hqueue->cur[level] < hqueue->qsize);
#endif
	hqueue->queue[hqueue->cur[level]++] = newidx;
}

inline uint32 hqueue_pop(HQueue* hqueue)
{
	return hqueue->queue[--hqueue->cur[hqueue->min_level]];
}

inline void hqueue_find_min_level(HQueue* hqueue)
{
	while (hqueue->bottom[hqueue->min_level] == hqueue->cur[hqueue->min_level])
		hqueue->min_level++;
}

typedef struct AlphaNode
{
	uint32 area;
	uint8 level;  /* alpha of flat zone */
	double sumPix;
	pixel minPix;
	pixel maxPix;
	uint32 parentidx;
} AlphaNode;

typedef struct AlphaTree
{
	uint32 maxSize;
	uint32 curSize;
	uint32 height, width, channel;
	AlphaNode* node;
	uint32* parentAry;
} AlphaTree;


inline void connectPix2Node(uint32* parentAry, uint32 pidx, uint8 pix_val, AlphaNode* pNode, uint32 iNode)
{
	parentAry[pidx] = iNode;
	pNode->area++;
	pNode->maxPix = max(pNode->maxPix, pix_val);
	pNode->minPix = min(pNode->minPix, pix_val);
	pNode->sumPix += pix_val;
}

inline void connectNode2Node(AlphaNode* pPar, uint32 iPar, AlphaNode* pNode)
{
	pNode->parentidx = iPar;
	pPar->area += pNode->area;
	pPar->maxPix = max(pNode->maxPix, pPar->maxPix);
	pPar->minPix = min(pNode->minPix, pPar->minPix);
	pPar->sumPix += pNode->sumPix;
}

void compute_dimg(uint8* dimg, uint32* dhist, uint8* img, uint32 height, uint32 width, uint32 channel)
{

	uint32 dimgidx, imgidx, stride_w = width, i, j, ch;
	if (channel == 1)
	{
		imgidx = dimgidx = 0;
		for (i = 0; i < height - 1; i++)
		{
			for (j = 0; j < width - 1; j++)
			{
				dimg[dimgidx] = (uint8)abs((int)img[imgidx + stride_w] - (int)img[imgidx]);
				dhist[dimg[dimgidx++]]++;
				dimg[dimgidx] = (uint8)abs((int)img[imgidx + 1] - (int)img[imgidx]);
				dhist[dimg[dimgidx++]]++;
				imgidx++;
			}
			dimg[dimgidx] = (uint8)abs((int)img[imgidx + stride_w] - (int)img[imgidx]);
			dhist[dimg[dimgidx++]]++;
			dimgidx++;
			imgidx++;
		}
		for (j = 0; j < width - 1; j++)
		{
			dimgidx++;
			dimg[dimgidx] = (uint8)abs((int)img[imgidx + 1] - (int)img[imgidx]);
			dhist[dimg[dimgidx++]]++;
			imgidx++;
		}
	}
	else
	{
		for (ch = 0; ch < channel; ch++)
		{
			imgidx = dimgidx = 0;
			for (i = 0; i < height - 1; i++)
			{
				for (j = 0; j < width - 1; j++)
				{
					dimg[dimgidx++] = max(dimg[dimgidx], (uint8)abs((int)img[imgidx + stride_w] - (int)img[imgidx])); // use Lmax dissim
					if (ch == channel - 1)
						dhist[dimg[dimgidx - 1]]++;
					dimg[dimgidx++] = max(dimg[dimgidx], (uint8)abs((int)img[imgidx + 1] - (int)img[imgidx]));
					if (ch == channel - 1)
						dhist[dimg[dimgidx - 1]]++;
					imgidx++;
				}
				dimg[dimgidx++] = max(dimg[dimgidx], (uint8)abs((int)img[imgidx + stride_w] - (int)img[imgidx]));
				if (ch == channel - 1)
					dhist[dimg[dimgidx - 1]]++;
				dimgidx++;
				imgidx++;
			}
			for (j = 0; j < width - 1; j++)
			{
				dimgidx++;
				dimg[dimgidx++] = max(dimg[dimgidx], (uint8)abs((int)img[imgidx + 1] - (int)img[imgidx]));
				if (ch == channel - 1)
					dhist[dimg[dimgidx - 1]]++;
				imgidx++;
			}
			img += width * height;
		}
	}
}
inline uint32 NewAlphaNode(AlphaTree* tree, uint8 level)
{
	AlphaNode *pNew = tree->node + tree->curSize;
	pNew->level = level;
	pNew->minPix = (pixel)-1;
	pNew->minPix = 0;
	pNew->sumPix = 0.0;
	pNew->parentidx = 0;
	pNew->area = 0;

	return tree->curSize++;
}

inline uint8 is_visited(uint8* isVisited, uint32 p)
{
	return (isVisited[p>>3] >> (p & 7)) & 1;
}

inline void visit(uint8* isVisited, uint32 p)
{
	isVisited[p >> 3] = isVisited[p >> 3] | (1 << (p & 7));
}


void Flood(AlphaTree* tree, uint8* img, uint32 height, uint32 width, uint32 channel)
{
	uint32 imgsize, dimgsize, nredges, max_level, current_level, next_level, x0, p, dissim;
	uint32 numlevels;
	HQueue* hqueue;
	uint32 *dhist;
	pixel *dimg;
	uint32 iChild, *levelroot;
	uint8 *isVisited;
	AlphaNode *pNode;
	uint32 *pParentAry;

	imgsize = width * height;
	nredges = width * (height - 1) + (width - 1) * height;
	dimgsize = 2 * width * height; //To make indexing easier
	numlevels = 1 << (8 * sizeof(uint8));

	//tmp_mem_size = imgsize * ISVISITED_ELEMENT_SIZE + (nredges + 1) * (HQUEUE_ELEMENT_SIZE)+dimgsize * (DIMG_ELEMENT_SIZE)+ 
	//numlevels * (LEVELROOT_ELEMENT_SIZE + DHIST_ELEMENT_SIZE + sizeof(hqueue_index)) + sizeof(hqueue_index) + LEVELROOT_ELEMENT_SIZE;


	dhist = (uint32*)malloc((size_t)numlevels * sizeof(uint32));
	dimg = (pixel*)malloc((size_t)dimgsize * sizeof(pixel));
	levelroot = (uint32*)malloc((uint32)(numlevels + 1) * LEVELROOT_ELEMENT_SIZE);
	isVisited = (uint8*)malloc((size_t)(imgsize >> 3));
	for (p = 0; p < numlevels; p++)
		levelroot[p] = NULL_LEVELROOT;
	memset(dhist, 0, (size_t)numlevels * sizeof(uint32));
	memset(isVisited, 0, (size_t)(imgsize >> 3));

	max_level = (uint8)(numlevels - 1);
	
	compute_dimg(dimg, dhist, img, height, width, channel);
	dhist[max_level]++;
	hqueue = hqueue_new(nredges + 1, dhist, numlevels);

	tree->height = height;
	tree->width = width;
	tree->channel = channel;
	tree->curSize = 0;
	tree->maxSize = imgsize + 1;//tree size estimation
	tree->parentAry = (uint32*)malloc((size_t)imgsize * sizeof(uint32));
	tree->node = (AlphaNode*)malloc((size_t)tree->maxSize * sizeof(AlphaNode));
	pNode = tree->node;
	pParentAry = tree->parentAry;

	levelroot[max_level + 1] = NewAlphaNode(tree, (uint8)max_level);
	pNode[levelroot[max_level + 1]].parentidx = levelroot[max_level + 1];

	current_level = max_level;
	x0 = imgsize >> 1;
	hqueue_push(hqueue, x0, current_level);

	iChild = levelroot[max_level + 1];
	while (current_level <= max_level)
	{
		while (hqueue->min_level <= current_level)
		{
			p = hqueue_pop(hqueue);
			if (is_visited(isVisited, p))
			{
				hqueue_find_min_level(hqueue);
				continue;
			}
			visit(isVisited, p);
#if !HQUEUE_COST_AMORTIZE
			hqueue_find_min_level();
#endif

			if (LEFT_AVAIL(p, width) && !is_visited(isVisited, p - 1))
			{
				dissim = (uint32)dimg[dimg_idx_h(p - 1)];
				hqueue_push(hqueue, p - 1, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (RIGHT_AVAIL(p, width) && !is_visited(isVisited, p + 1))
			{
				dissim = (uint32)dimg[dimg_idx_h(p)];
				hqueue_push(hqueue, p + 1, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (UP_AVAIL(p, width) && !is_visited(isVisited, p - width))
			{
				dissim = (uint32)dimg[dimg_idx_v(p - width)];
				hqueue_push(hqueue, p - width, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (DOWN_AVAIL(p, width, imgsize) && !is_visited(isVisited, p + width))
			{
				dissim = (uint32)dimg[dimg_idx_v(p)];
				hqueue_push(hqueue, p + width, dissim);
				if (levelroot[dissim] == NULL_LEVELROOT)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}

			if (current_level > hqueue->min_level)
				current_level = hqueue->min_level;
#if HQUEUE_COST_AMORTIZE
			else
				hqueue_find_min_level(hqueue);
#endif

#if DELAYED_ANODE_ALLOC
			if (levelroot[current_level] == ANODE_CANDIDATE)
				levelroot[current_level] = NewAlphaNode(tree, (uint8)current_level);
#endif
			connectPix2Node(pParentAry, p, img[p], pNode + levelroot[current_level], levelroot[current_level]);
		}
//		if(tree->curSize > 22051838 && (tree->curSize))
	//		printf("curSize: %d\n",tree->curSize);
		//Redundant node removal
		if (pNode[iChild].parentidx == levelroot[current_level] &&
			pNode[levelroot[current_level]].area == pNode[iChild].area)
		{
			levelroot[current_level] = iChild;
			tree->curSize--;
			
			memset((uint8*)(pNode + tree->curSize), 0, sizeof(AlphaNode));
		}

		next_level = current_level + 1;
		while (next_level <= max_level && (levelroot[next_level] == NULL_LEVELROOT))
			next_level++;
		if (levelroot[next_level] == ANODE_CANDIDATE)
			levelroot[next_level] = NewAlphaNode(tree, (uint8)next_level);
		connectNode2Node(pNode + levelroot[next_level], levelroot[next_level], pNode + levelroot[current_level]);
		iChild = levelroot[current_level];
		levelroot[current_level] = NULL_LEVELROOT;
		current_level = next_level;
	}
	hqueue_free(hqueue);
	free(dhist);
	free(dimg);
	free(levelroot);
	free(isVisited);
}


void BuildAlphaTree(AlphaTree* tree, uint8 *img, uint32 height, uint32 width, uint32 channel)
{
	Flood(tree, img, height, width, channel);
}

void DeleteAlphaTree(AlphaTree* tree)
{
	free(tree->parentAry);
	free(tree->node);
	free(tree);
}

int main(int argc, char **argv)
{	
	cv::String str1("C:/jwryu/RUG/2018/AlphaTree/imgdata/Colour_imgs/44767133841_7b7ec3ccf6_o.jpg");
	cv::Mat img = imread(str1, cv::IMREAD_ANYCOLOR);
	AlphaTree tree;

		//	cv::String str("C:/jwryu/RUG/2018/AlphaTree/imgdata/Colour_imgs/16578511714_6eaef1c5bb_o.jpg");
//	cv::Mat img = imread(str, CV_LOAD_IMAGE_COLOR);
//	cv::String str("C:/jwryu/RUG/2018/AlphaTree/imgdata/Aerial_Grey/s-gravenhage_33696704805_o.jpg");
//	cv::Mat img = imread(str, CV_LOAD_IMAGE_GRAYSCALE);
	/*
	uint32 height, width, channel;
	
	char fname[] = "C:/jwryu/RUG/2018/AlphaTree/imgdata/remote_sensing_img_8bit_8281x8185.raw";
	height = 8281;
	width = 8185;
	channel = 1;
	uint8 *img;
	FILE *fp;

	img = (uint8*)malloc(height*width*sizeof(uint8));
	fopen_s(&fp, fname, "r");
	fread(img, sizeof(uint8), height*width*channel, fp);
	fclose(fp);
	memcpy(img, img1.data, height*width*channel*sizeof(uint8));
	*/
	//uint8 testimg[9] = { 4, 4, 1, 4, 2, 2, 1, 2, 0 };

	//AlphaTree aTree;
	//	struct timeval stop, start;
	clock_t start, stop;

//	printf("Image size: %dx%dx%d", img.rows, img.cols, img.channels());
	//	gettimeofday(&start, NULL);
	start = clock();
	BuildAlphaTree(&tree, (uint8*)img.data, img.rows, img.cols, img.channels());
//	BuildAlphaTree(&tree, (uint8*)img1.data, height, width, channel);
	stop = clock();
	//	gettimeofday(&stop, NULL);
	//		namedWindow("disp", WINDOW_AUTOSIZE); // Create a window for display.
		//	imshow("disp", img);                // Show our image inside it.
			//waitKey(0);
	printf("Time Elapsed: %f", (double)(stop - start) / 1000.0);
	getc(stdin);
//	free(img);
	img.release();
	return 0;
}