/*
Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#ifndef ECL_GRAPH
#define ECL_GRAPH

#include <cstdlib>
#include <cstdio>

struct ECLgraph
{
    int nodes;
    int edges1;
    int *nindex;
    int *nlist;
    int *eweight;
};

ECLgraph readECLgraph(const char *const fname)
{
    ECLgraph g;
    int cnt;

    FILE *f = fopen(fname, "rb");
    if (f == NULL)
    {
        fprintf(stderr, "ERROR: could not open file %s\n\n", fname);
        exit(-1);
    }
    cnt = fread(&g.nodes, sizeof(g.nodes), 1, f);
    if (cnt != 1)
    {
        fprintf(stderr, "ERROR: failed to read nodes\n\n");
        exit(-1);
    }
    cnt = fread(&g.edges1, sizeof(g.edges1), 1, f);
    if (cnt != 1)
    {
        fprintf(stderr, "ERROR: failed to read edges1\n\n");
        exit(-1);
    }
    if ((g.nodes < 1) || (g.edges1 < 0))
    {
        fprintf(stderr, "ERROR: node or edge count too low\n\n");
        exit(-1);
    }

    g.nindex = (int *)malloc((g.nodes + 1) * sizeof(g.nindex[0]));
    g.nlist = (int *)malloc(g.edges1 * sizeof(g.nlist[0]));
    g.eweight = (int *)malloc(g.edges1 * sizeof(g.eweight[0]));
    if ((g.nindex == NULL) || (g.nlist == NULL) || (g.eweight == NULL))
    {
        fprintf(stderr, "ERROR: memory allocation failed\n\n");
        exit(-1);
    }

    cnt = fread(g.nindex, sizeof(g.nindex[0]), g.nodes + 1, f);
    if (cnt != g.nodes + 1)
    {
        fprintf(stderr, "ERROR: failed to read neighbor index list\n\n");
        exit(-1);
    }
    cnt = fread(g.nlist, sizeof(g.nlist[0]), g.edges1, f);
    if (cnt != g.edges1)
    {
        fprintf(stderr, "ERROR: failed to read neighbor list\n\n");
        exit(-1);
    }
    cnt = fread(g.eweight, sizeof(g.eweight[0]), g.edges1, f);
    if (cnt == 0)
    {
        free(g.eweight);
        g.eweight = NULL;
    }
    else
    {
        if (cnt != g.edges1)
        {
            fprintf(stderr, "ERROR: failed to read edge weights\n\n");
            exit(-1);
        }
    }
    fclose(f);

    return g;
}

void writeECLgraph(const ECLgraph g, const char *const fname)
{
    if ((g.nodes < 1) || (g.edges1 < 0))
    {
        fprintf(stderr, "ERROR: node or edge count too low\n\n");
        exit(-1);
    }
    int cnt;
    FILE *f = fopen(fname, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "ERROR: could not open file %s\n\n", fname);
        exit(-1);
    }
    cnt = fwrite(&g.nodes, sizeof(g.nodes), 1, f);
    if (cnt != 1)
    {
        fprintf(stderr, "ERROR: failed to write nodes\n\n");
        exit(-1);
    }
    cnt = fwrite(&g.edges1, sizeof(g.edges1), 1, f);
    if (cnt != 1)
    {
        fprintf(stderr, "ERROR: failed to write edges1\n\n");
        exit(-1);
    }

    cnt = fwrite(g.nindex, sizeof(g.nindex[0]), g.nodes + 1, f);
    if (cnt != g.nodes + 1)
    {
        fprintf(stderr, "ERROR: failed to write neighbor index list\n\n");
        exit(-1);
    }
    cnt = fwrite(g.nlist, sizeof(g.nlist[0]), g.edges1, f);
    if (cnt != g.edges1)
    {
        fprintf(stderr, "ERROR: failed to write neighbor list\n\n");
        exit(-1);
    }
    if (g.eweight != NULL)
    {
        cnt = fwrite(g.eweight, sizeof(g.eweight[0]), g.edges1, f);
        if (cnt != g.edges1)
        {
            fprintf(stderr, "ERROR: failed to write edge weights\n\n");
            exit(-1);
        }
    }
    fclose(f);
}

void freeECLgraph(ECLgraph &g)
{
    if (g.nindex != NULL)
        free(g.nindex);
    if (g.nlist != NULL)
        free(g.nlist);
    if (g.eweight != NULL)
        free(g.eweight);
    g.nindex = NULL;
    g.nlist = NULL;
    g.eweight = NULL;
}

#endif
