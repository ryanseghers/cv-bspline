#include "BezierUtil.h"

namespace CvImageDeform
{
    const float THIRDS[] = { 0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };

    float bezierPolyTerm(int i, float t) 
    {
        float u = 1 - t;
        switch (i) 
        {
        case 0: return u*u*u;
        case 1: return 3*t*u*u;
        case 2: return 3*t*t*u;
        case 3: return t*t*t;
        }
        return 0;
    }
}
