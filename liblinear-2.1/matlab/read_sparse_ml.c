#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include "mex.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif 

void exit_with_help()
{
	mexPrintf(
	"Usage: [label_vector, instance_matrix, label_map] = read_sparse_ml(fname);\n"
	);
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void read_problem(const char *filename, mxArray *plhs[])
{
	int elements, max_index, min_index, i, j, k, yk;
	int n_labels, max_n_label;
	FILE *fp = fopen(filename,"r");
	int l = 0;
	mwIndex *ir, *jc;
	mwIndex *y_ir, *y_jc;
	double *labels, *samples, *map;

	int nr_label = 0;
	int max_nr_label = 16;
	int *label = (int *)malloc(sizeof(int) * max_nr_label);

	int *instance_labels;
	
	if(fp == NULL)
	{
		mexPrintf("can't open input file %s\n",filename);
		fake_answer(plhs);
		return;
	}

	max_index = 0;
	min_index = 1; 
	elements = 0;
	n_labels = 0;
	max_n_label = 0;
	while(1)
	{
		int index, c;
		double value;

		index = 0;
		do {
			int this_label;

			c = getc(fp);
			if(c==EOF) goto eof;
			if(c=='\n' || c==' ') break;
			ungetc(c,fp);

			fscanf(fp,"%lf",&value);

			/* XXX classification only */
			this_label = (int)value;
			for(j=0;j<nr_label;j++)
			{
				if(this_label == label[j])
					break;
			}
			if(j == nr_label)
			{
				if(nr_label == max_nr_label)
				{
					max_nr_label *= 2;
					label = (int *)realloc(label, max_nr_label * sizeof(int));
				}
				label[nr_label] = this_label;
				++nr_label;
			}

			n_labels++;
			index++;

			c = getc(fp);
		} while(c == ',');
		ungetc(c,fp);

		if(index > max_n_label)
			max_n_label = index;

		index = 0;
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out;
				if(c==EOF) goto eof;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&index, &value);
			if (index < min_index)
				min_index = index;
			elements++;
		}	
out:
		if(index > max_index)
			max_index = index;
		l++;
	}
eof:

	rewind(fp);

	plhs[0] = mxCreateSparse(nr_label, l, n_labels, mxREAL);
	if (min_index <= 0)
		plhs[1] = mxCreateSparse(max_index-min_index+1, l, elements, mxREAL);
	else
		plhs[1] = mxCreateSparse(max_index, l, elements, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(nr_label, 1, mxREAL);

	labels = mxGetPr(plhs[0]);
	y_ir = mxGetIr(plhs[0]);
	y_jc = mxGetJc(plhs[0]);

	samples = mxGetPr(plhs[1]);
	ir = mxGetIr(plhs[1]);
	jc = mxGetJc(plhs[1]);

	map = mxGetPr(plhs[2]);
	for(j=0;j<nr_label;j++)
		map[j] = label[j];

	k=0;
	yk=0;

	instance_labels = (int *) malloc(sizeof(int)*nr_label);
	for(i=0;i<l;i++)
	{
		int c, this_label;
		double value;

		memset(instance_labels, 0, sizeof(int)*nr_label);

		y_jc[i] = yk;

		do {
			c = getc(fp);
			if(c=='\n' || c==' ') break;
			ungetc(c,fp);

			fscanf(fp,"%lf",&value);

			/* XXX binary search or hash */
			this_label = (int)value;
			for(j=0;j<nr_label;j++)
			{
				if(this_label == label[j])
					break;
			}

			if(instance_labels[j] == 1)
				mexPrintf("ERROR: duplicate label %d in %d-th instance\n", i+1, label[j]);
			else
				instance_labels[j] = 1;

			y_ir[yk] = j;
			labels[yk] = 1;
			yk++;

			c = getc(fp);
		} while(c == ',');
		ungetc(c,fp);

		jc[i] = k;

		while(1)
		{
			int c, index;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&index,&samples[k]);
			ir[k] = index - min_index; 
			++k;
		}	
out2:
		;
	}

	free(instance_labels);

	jc[l] = k;
	y_jc[l] = yk;

	fclose(fp);

	{
		mxArray *rhs[1], *lhs[1];
		rhs[0] = plhs[0];
		if(mexCallMATLAB(1, lhs, 1, rhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose labels\n");
			return;
		}
		plhs[0] = lhs[0];
	}

	{
		mxArray *rhs[1], *lhs[1];
		rhs[0] = plhs[1];
		if(mexCallMATLAB(1, lhs, 1, rhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose problem\n");
			return;
		}
		plhs[1] = lhs[0];
	}

	free(label);
}

void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	if(nrhs == 1)
	{
		char filename[256];

		mxGetString(prhs[0], filename, mxGetN(prhs[0]) + 1);

		if(filename == NULL)
		{
			mexPrintf("Error: filename is NULL\n");
			return;
		}

		read_problem(filename, plhs);
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}
