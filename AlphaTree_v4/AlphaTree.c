#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <string.h>
//#include <opencv2/opencv.hpp>
#include <time.h>

#define DEBUG 0
//#define max(a,b) (a)>(b)?(a):(b)
//#define min(a,b) (a)>(b)?(b):(a)

#define DELAYED_ANODE_ALLOC		1
#define HQUEUE_COST_AMORTIZE	1

#define ANODE_CANDIDATE		(AlphaNode*)-1

#define DIMGIDX_V(pidx) ((pidx)<<1)
#define DIMGIDX_H(pidx) ((pidx)<<1)+1

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


typedef struct hqueue_index
{
	uint32 bottom;
	uint32 cur;
}hqueue_index;

typedef struct HQueue
{
	uint32 *queue;
	hqueue_index *qidx;
	uint64 qsize;
	uint32 min_level, max_level;
}HQueue;


HQueue* newHQueue(uint64 qsize, uint32 *dhist, uint32 dhistsize)
{
	HQueue* hqueue = (HQueue*)malloc(sizeof(HQueue));
	hqueue->queue = (uint32*)malloc((size_t)qsize * sizeof(uint32));
	hqueue->qidx = (hqueue_index*)malloc((size_t)(dhistsize + 1) * sizeof(hqueue_index));

	hqueue->qsize = qsize;
	hqueue->min_level = hqueue->max_level = dhistsize;

	
	int sum_hist = 0;
	for (uint32 i = 0; i < dhistsize; i++)
	{
		hqueue->qidx[i].bottom = hqueue->qidx[i].cur = sum_hist;
		sum_hist += dhist[i];
	}
	hqueue->qidx[dhistsize].bottom = 0;
	hqueue->qidx[dhistsize].cur = 1;

	return hqueue;
}

void deleteHQueue(HQueue* hqueue)
{
	free(hqueue->queue);
	free(hqueue->qidx);
	free(hqueue);
}

inline uint8 hqueue_is_empty(HQueue* hqueue)
{
	return hqueue->min_level == hqueue->max_level;
}

inline void hqueue_push(HQueue* hqueue, uint32 newidx, uint32 level)
{
	hqueue->min_level = min(level, hqueue->min_level);
#if DEBUG
	assert(level < hqueue->max_level);
	assert(hqueue->qidx[level].cur < hqueue->qsize);
#endif
	hqueue->queue[hqueue->qidx[level].cur++] = newidx;
}

inline uint32 hqueue_pop(HQueue* hqueue)
{
	return hqueue->queue[hqueue->qidx[hqueue->min_level].bottom++];
}

inline void hqueue_find_min_level(HQueue* hqueue)
{
	while (hqueue->qidx[hqueue->min_level].bottom == hqueue->qidx[hqueue->min_level].cur)
		hqueue->min_level++;
}


typedef struct AlphaNode
{
	uint32 area;
	uint8 level;  /* alpha of flat zone */
	double sumPix;
	pixel minPix;
	pixel maxPix;
	struct AlphaNode *parent;
} AlphaNode;

typedef struct AlphaTree
{

	uint32 maxSize;
	uint32 curSize;
	uint32 height, width, channel;
	AlphaNode* node;
	AlphaNode** parentAry;
} AlphaTree;


inline void connectPix2Node(AlphaTree* tree, uint32 pidx, uint8 pix_val, AlphaNode* pNode)
{
	tree->parentAry[pidx] = pNode;
	pNode->area++;
	pNode->maxPix = max(pNode->maxPix, pix_val);
	pNode->minPix = min(pNode->minPix, pix_val);
	pNode->sumPix += pix_val;
}

inline void connectNode2Node(AlphaTree* tree, AlphaNode* pPar, AlphaNode* pNode)
{
	pNode->parent = pPar;
	pPar->area += pNode->area;
	pPar->maxPix = max(pNode->maxPix, pPar->maxPix);
	pPar->minPix = min(pNode->minPix, pPar->minPix);
	pPar->sumPix += pNode->sumPix;
}

void compute_dimg(uint8* dimg, uint32* dhist, uint8* img, uint32 height, uint32 width, uint32 channel)
{

	uint32 dimgidx, imgidx, stride_w = width;
	if (channel == 1)
	{
		imgidx = dimgidx = 0;
		for (uint32 i = 0; i < height - 1; i++)
		{
			for (uint32 j = 0; j < width - 1; j++)
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
		for (uint32 j = 0; j < width - 1; j++)
		{
			dimgidx++;
			dimg[dimgidx] = (uint8)abs((int)img[imgidx + 1] - (int)img[imgidx]);
			dhist[dimg[dimgidx++]]++;
			imgidx++;
		}
	}
	else
	{
		for (uint32 ch = 0; ch < channel; ch++)
		{
			imgidx = dimgidx = 0;
			for (uint32 i = 0; i < height - 1; i++)
			{
				for (uint32 j = 0; j < width - 1; j++)
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
			for (uint32 j = 0; j < width - 1; j++)
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
inline AlphaNode* NewAlphaNode(AlphaTree* tree, uint8 level)
{
	AlphaNode *pNew = tree->node + tree->curSize++;
	pNew->level = level;
	pNew->minPix = (pixel)-1;
	pNew->minPix = 0;
	pNew->sumPix = 0.0;
	pNew->parent = NULL;
	pNew->area = 0;

	return pNew;
}


void Flood(AlphaTree* tree, uint8* img, uint32 height, uint32 width, uint32 channel)
{
	uint32 imgsize, dimgsize, nredges, max_level, current_level, next_level, x0, p, dissim;
	uint32 numlevels;
	HQueue* hqueue;
	uint32 *dhist;
	pixel *dimg;
	AlphaNode *pChild, **levelroot;
	uint8 *isVisited;

	imgsize = width * height;
	nredges = width * (height - 1) + (width - 1) * height;
	dimgsize = 2 * width * height; //To make indexing easier
	numlevels = 1 << (8 * sizeof(uint8));

	//tmp_mem_size = imgsize * ISVISITED_ELEMENT_SIZE + (nredges + 1) * (HQUEUE_ELEMENT_SIZE)+dimgsize * (DIMG_ELEMENT_SIZE)+ 
	//numlevels * (LEVELROOT_ELEMENT_SIZE + DHIST_ELEMENT_SIZE + sizeof(hqueue_index)) + sizeof(hqueue_index) + LEVELROOT_ELEMENT_SIZE;
	

	dhist = (uint32*)malloc((size_t)numlevels * sizeof(uint32));
	dimg = (pixel*)malloc((size_t)dimgsize * sizeof(pixel));
	levelroot = (AlphaNode**)malloc((size_t)(numlevels + 1) * LEVELROOT_ELEMENT_SIZE);
	isVisited = (uint8*)malloc((size_t)imgsize * ISVISITED_ELEMENT_SIZE);
	memset(dhist, 0, (size_t)numlevels * sizeof(uint32));
	memset(levelroot, 0, (size_t)(numlevels + 1) * LEVELROOT_ELEMENT_SIZE);
	memset(isVisited, 0, (size_t)imgsize * ISVISITED_ELEMENT_SIZE);


	max_level = (uint8)(numlevels - 1);
	
	compute_dimg(dimg, dhist, img, height, width, channel);
	dhist[max_level]++;
	hqueue = newHQueue(nredges + 1, dhist, numlevels);

	tree->height = height;
	tree->width = width;
	tree->channel = channel;
	tree->curSize = 0;
	tree->maxSize = imgsize + 1;//tree size estimation
	tree->parentAry = (AlphaNode**)malloc((size_t)imgsize * sizeof(AlphaNode*));
	tree->node = (AlphaNode*)malloc((size_t)tree->maxSize * sizeof(AlphaNode));
	//memset(tree->node, 0, (size_t)tree->maxSize * sizeof(AlphaNode));

	levelroot[max_level + 1] = NewAlphaNode(tree, (uint8)max_level);
	levelroot[max_level + 1]->parent = levelroot[max_level + 1];

	current_level = max_level;
	x0 = imgsize >> 1;
	hqueue_push(hqueue, x0, current_level);

	pChild = levelroot[max_level + 1];
	while (current_level <= max_level)
	{
		while (hqueue->min_level <= current_level)
		{
			p = hqueue_pop(hqueue);
			if (isVisited[p])
			{
				hqueue_find_min_level(hqueue);
				continue;
			}
			isVisited[p] = 1;
#if !HQUEUE_COST_AMORTIZE
			hqueue_find_min_level();
#endif

			if (LEFT_AVAIL(p, width) && !isVisited[p - 1])
			{
				dissim = (uint32)dimg[DIMGIDX_H(p - 1)];
				hqueue_push(hqueue, p - 1, dissim);
				if (levelroot[dissim] == NULL)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (RIGHT_AVAIL(p, width) && !isVisited[p + 1])
			{
				dissim = (uint32)dimg[DIMGIDX_H(p)];
				hqueue_push(hqueue, p + 1, dissim);
				if (levelroot[dissim] == NULL)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (UP_AVAIL(p, width) && !isVisited[p - width])
			{
				dissim = (uint32)dimg[DIMGIDX_V(p - width)];
				hqueue_push(hqueue, p - width, dissim);
				if (levelroot[dissim] == NULL)
#if DELAYED_ANODE_ALLOC
					levelroot[dissim] = ANODE_CANDIDATE;
#else
					levelroot[dissim] = NewAlphaNode(tree, (uint8)dissim);
#endif
			}
			if (DOWN_AVAIL(p, width, imgsize) && !isVisited[p + width])
			{
				dissim = (uint32)dimg[DIMGIDX_V(p)];
				hqueue_push(hqueue, p + width, dissim);
				if (levelroot[dissim] == NULL)
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
			connectPix2Node(tree, p, img[p], levelroot[current_level]);
		}

		//Redundant node removal
		if (pChild->parent == levelroot[current_level] &&
			levelroot[current_level]->area == pChild->area)
		{
			levelroot[current_level] = pChild;
			tree->curSize--;
			//memset((uint8*)(tree->node + tree->curSize), 0, sizeof(AlphaNode));
		}

		next_level = current_level + 1;
		while (next_level <= max_level && (levelroot[next_level] == NULL))
			next_level++;
		if (levelroot[next_level] == ANODE_CANDIDATE)
			levelroot[next_level] = NewAlphaNode(tree, (uint8)next_level);
		connectNode2Node(tree, levelroot[next_level], levelroot[current_level]);
		pChild = levelroot[current_level];
		levelroot[current_level] = NULL;
		current_level = next_level;
	}

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
	//	String str("C:/jwryu/RUG/2018/AlphaTree/imgdata/Colour_imgs/16578511714_6eaef1c5bb_o.jpg");
	//	Mat img = imread(str, CV_LOAD_IMAGE_COLOR);

//	String str("C:/jwryu/RUG/2018/AlphaTree/imgdata/remote_sensing_img.pgm");
//	Mat img = imread(str, CV_LOAD_IMAGE_GRAYSCALE);

	char *fname = "C:/jwryu/RUG/2018/AlphaTree/imgdata/remote_sensing_img_8bit_8281x8185.raw";
	AlphaTree tree;
	uint8 *img;
	uint32 height, width, channel;
	FILE *fp;

	height = 8281;
	width = 8185;
	channel = 1;
	img = (uint8*)malloc(height*width*sizeof(uint8));
	fopen_s(&fp, fname, "r");
	fread(img, sizeof(uint8), height*width*channel, fp);
	fclose(fp);
	//uint8 testimg[9] = { 4, 4, 1, 4, 2, 2, 1, 2, 0 };

	//AlphaTree aTree;
	//	struct timeval stop, start;
	clock_t start, stop;

//	printf("Image size: %dx%dx%d", img.rows, img.cols, img.channels());
	//	gettimeofday(&start, NULL);
	start = clock();
	BuildAlphaTree(&tree, img, height, width, channel);
//	aTree.BuildAlphaTree((uint8*)img.data, img.rows, img.cols, img.channels());
	stop = clock();
	//	gettimeofday(&stop, NULL);
	//		namedWindow("disp", WINDOW_AUTOSIZE); // Create a window for display.
		//	imshow("disp", img);                // Show our image inside it.
			//waitKey(0);
	printf("Time Elapsed: %f", (double)(stop - start) / 1000.0);
	getc(stdin);
	free(img);
//	img.release();
	return 0;
}