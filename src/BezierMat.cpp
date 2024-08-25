#ifdef _WIN32
#include <immintrin.h>
#endif

#include "BezierMat.h"
#include "BezierUtil.h"

using namespace std;

namespace CvImageDeform
{
    cv::Point3f evalBezierSurfaceCubicPointMat(const cv::Mat& controlPointsZs, float u, float v)
    {
        cv::Point3f point(0, 0, 0);

        for (int i = 0; i < 4; ++i)
        {
            float y = THIRDS[i];

            for (int j = 0; j < 4; ++j)
            {
                float x = THIRDS[j];
                cv::Point3f cp(x, y, controlPointsZs.at<float>(i, j));
                point += bezierPolyTerm(i, u) * bezierPolyTerm(j, v) * cp;
            }
        }

        return point;
    }

    cv::Point3f evalBezierSurfaceCubicPointMatUnrolled(const cv::Mat& controlPointsZs, float u, float v)
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float u1 = 1.0f - u;
        float v1 = 1.0f - v;
        float v13 = v1 * v1 * v1;
        float v12 = 3 * v * v1 * v1;
        float v21 = 3 * v * v * v1;
        float v3 = v * v * v;

        cv::Point3f point(0, 0, 0);
        cv::Point3f cp(0, 0, 0);

        for (int i = 0; i < 4; ++i)
        {
            float y = THIRDS[i];
            cp.y = y;
            float biu = bezierPolyTerm(i, u);

            // Unrolled using points
            // j = 0
            float bjv = v13;
            cp.x = 0.0f;
            cp.z = controlPointsZs.at<float>(i, 0);
            point += biu * bjv * cp;

            bjv = v12;
            cp.x = THIRDS[1];
            cp.z = controlPointsZs.at<float>(i, 1);
            point += biu * bjv * cp;

            bjv = v21;
            cp.x = THIRDS[2];
            cp.z = controlPointsZs.at<float>(i, 2);
            point += biu * bjv * cp;

            bjv = v3;
            cp.x = 1.0f;
            cp.z = controlPointsZs.at<float>(i, 3);
            point += biu * bjv * cp;

        }

        return point;
    }

    cv::Point3f evalBezierSurfaceCubicPointMatUnrolled2(const cv::Mat& controlPointsZs, float u, float v)
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float u1 = 1.0f - u;
        float v1 = 1.0f - v;
        float v13 = v1 * v1 * v1;
        float v12 = 3 * v * v1 * v1;
        float v21 = 3 * v * v * v1;
        float v3 = v * v * v;

        for (int i = 0; i < 4; ++i)
        {
            const float* pz = controlPointsZs.ptr<float>(i, 0);
            float tx;
            float ty = THIRDS[i];
            float biu = bezierPolyTerm(i, u);

            float bjv = v13;
            //tx = 0.0f;
            float tz = pz[0];
            //x += biu * bjv * tx;
            y += biu * bjv * ty;
            z += biu * bjv * tz;

            bjv = v12;
            tx = THIRDS[1];
            tz = pz[1];
            x += biu * bjv * tx;
            y += biu * bjv * ty;
            z += biu * bjv * tz;

            bjv = v21;
            tx = THIRDS[2];
            tz = pz[2];
            x += biu * bjv * tx;
            y += biu * bjv * ty;
            z += biu * bjv * tz;

            bjv = v3;
            //tx = 1.0f;
            tz = pz[3];
            x += biu * bjv; // * tx
            y += biu * bjv * ty;
            z += biu * bjv * tz;
        }

        return cv::Point3f(x, y, z);
    }

#ifdef _WIN32
    cv::Point3f evalBezierSurfaceCubicPointMatSimd(const cv::Mat& controlPointsZs, float u, float v)
    {
        __m256 xsums = _mm256_set1_ps(0.0f);
        __m256 ysums = _mm256_set1_ps(0.0f);
        __m256 zsums = _mm256_set1_ps(0.0f);

        float u1 = 1.0f - u;
        float v1 = 1.0f - v;
        float v13 = v1 * v1 * v1;
        float v12 = 3 * v * v1 * v1;
        float v21 = 3 * v * v * v1;
        float v3 = v * v * v;

        __m256 coeffs = _mm256_set_ps(v13, v12, v21, v3, 0.0f, 0.0f, 0.0f, 0.0f);
        __m256 thirds = _mm256_set_ps(THIRDS[0], THIRDS[1], THIRDS[2], THIRDS[3], 0.0f, 0.0f, 0.0f, 0.0f);

        for (int i = 0; i < 4; ++i)
        {
            const float* pz = controlPointsZs.ptr<float>(i, 0);
            __m256 zs = _mm256_set_ps(pz[0], pz[1], pz[2], pz[3], 0.0f, 0.0f, 0.0f, 0.0f);
            float biu = bezierPolyTerm(i, u);
            __m256 bius = _mm256_set_ps(biu, biu, biu, biu, 0.0f, 0.0f, 0.0f, 0.0f);

            //x += biu * v13 * THIRDS[0];
            //y += biu * v13 * THIRDS[i];
            //z += biu * v13 * pz[0];

            //x += biu * v12 * THIRDS[1];
            //y += biu * v12 * THIRDS[i];
            //z += biu * v12 * pz[1];

            //x += biu * v21 * THIRDS[2];
            //y += biu * v21 * THIRDS[i];
            //z += biu * v21 * pz[2];

            //x += biu * v3 * THIRDS[3];
            //y += biu * v3 * THIRDS[i];
            //z += biu * v3 * pz[3];

            __m256 thirdsi = _mm256_set1_ps(THIRDS[i]);

            __m256 p1 = _mm256_mul_ps(bius, coeffs);
            __m256 px1 = _mm256_mul_ps(p1, thirds);
            __m256 py1 = _mm256_mul_ps(p1, thirdsi);
            __m256 pz1 = _mm256_mul_ps(p1, zs);

            xsums = _mm256_add_ps(xsums, px1);
            ysums = _mm256_add_ps(ysums, py1);
            zsums = _mm256_add_ps(zsums, pz1);
        }

        float x = xsums.m256_f32[4] + xsums.m256_f32[5] + xsums.m256_f32[6] + xsums.m256_f32[7];
        float y = ysums.m256_f32[4] + ysums.m256_f32[5] + ysums.m256_f32[6] + ysums.m256_f32[7];
        float z = zsums.m256_f32[4] + zsums.m256_f32[5] + zsums.m256_f32[6] + zsums.m256_f32[7];

        return cv::Point3f(x, y, z);
    }

    /**
     * @brief Horizontal sum floats.
     */
    float hsum_avx(__m256 x)
    {
        __m128 vlow  = _mm256_castps256_ps128(x);
        __m128 vhigh = _mm256_extractf128_ps(x, 1);
        vlow  = _mm_add_ps(vlow, vhigh);
        vhigh = _mm_movehl_ps(vhigh, vlow);
        vlow  = _mm_add_ps(vlow, vhigh);
        vhigh = _mm_shuffle_ps(vlow, vlow, 0x55);
        vlow  = _mm_add_ss(vlow, vhigh);
        return _mm_cvtss_f32(vlow);
    }

    cv::Point3f evalBezierSurfaceCubicPointMatSimd2(const cv::Mat& controlPointsZs, float u, float v)
    {
        __m256 xsums = _mm256_set1_ps(0.0f);
        __m256 ysums = _mm256_set1_ps(0.0f);
        __m256 zsums = _mm256_set1_ps(0.0f);

        float u1 = 1.0f - u;
        float v1 = 1.0f - v;
        float v13 = v1 * v1 * v1;
        float v12 = 3 * v * v1 * v1;
        float v21 = 3 * v * v * v1;
        float v3 = v * v * v;

        // Compute coeffs and biu's in one register
        // (v13, v12, v21, v3 and bius)
        __m256 uvs0 = _mm256_set_ps(1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 3.0f, 3.0f, 1.0f);
        __m256 uvs1 = _mm256_set_ps(v1, v, v, v, u1, u, u, u);
        __m256 uvs2 = _mm256_set_ps(v1, v1, v, v, u1, u1, u, u);
        __m256 uvs3 = _mm256_set_ps(v1, v1, v1, v, u1, u1, u1, u);
        __m256 uvs_and_bius = _mm256_mul_ps(uvs0, uvs1);
        uvs_and_bius = _mm256_mul_ps(uvs_and_bius, uvs2);
        uvs_and_bius = _mm256_mul_ps(uvs_and_bius, uvs3);

        __m256 coeffs = _mm256_set_ps(v13, v12, v21, v3, v13, v12, v21, v3);
        __m256 thirds = _mm256_set_ps(THIRDS[0], THIRDS[1], THIRDS[2], THIRDS[3], THIRDS[0], THIRDS[1], THIRDS[2], THIRDS[3]);

        // Unroll changed from 50 to 38 ms
        {
            int i1 = 0;
            int i2 = 1;

            const float* zi1 = controlPointsZs.ptr<float>(i1, 0);
            const float* zi2 = controlPointsZs.ptr<float>(i2, 0);
            __m256 zs = _mm256_set_ps(zi1[0], zi1[1], zi1[2], zi1[3], zi2[0], zi2[1], zi2[2], zi2[3]);

            //float biu1 = bezierPolyTerm(i1, u);
            //float biu2 = bezierPolyTerm(i2, u);
            float biu1 = u1 * u1 * u1;
            float biu2 = 3 * u * u1 * u1;
            __m256 bius = _mm256_set_ps(biu1, biu1, biu1, biu1, biu2, biu2, biu2, biu2);
            __m256 thirdsi = _mm256_set_ps(THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i2], THIRDS[i2], THIRDS[i2], THIRDS[i2]);

            __m256 p1 = _mm256_mul_ps(bius, coeffs);
            __m256 px1 = _mm256_mul_ps(p1, thirds);
            __m256 py1 = _mm256_mul_ps(p1, thirdsi);
            __m256 pz1 = _mm256_mul_ps(p1, zs);

            xsums = _mm256_add_ps(xsums, px1);
            ysums = _mm256_add_ps(ysums, py1);
            zsums = _mm256_add_ps(zsums, pz1);
        }

        {
            int i1 = 2;
            int i2 = 3;

            const float* zi1 = controlPointsZs.ptr<float>(i1, 0);
            const float* zi2 = controlPointsZs.ptr<float>(i2, 0);
            __m256 zs = _mm256_set_ps(zi1[0], zi1[1], zi1[2], zi1[3], zi2[0], zi2[1], zi2[2], zi2[3]);

            //float biu3 = bezierPolyTerm(i1, u);
            //float biu4 = bezierPolyTerm(i2, u);
            float biu3 = 3 * u * u * u1;
            float biu4 = u * u * u;
            __m256 bius = _mm256_set_ps(biu3, biu3, biu3, biu3, biu4, biu4, biu4, biu4);
            __m256 thirdsi = _mm256_set_ps(THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i2], THIRDS[i2], THIRDS[i2], THIRDS[i2]);

            __m256 p1 = _mm256_mul_ps(bius, coeffs);
            __m256 px1 = _mm256_mul_ps(p1, thirds);
            __m256 py1 = _mm256_mul_ps(p1, thirdsi);
            __m256 pz1 = _mm256_mul_ps(p1, zs);

            xsums = _mm256_add_ps(xsums, px1);
            ysums = _mm256_add_ps(ysums, py1);
            zsums = _mm256_add_ps(zsums, pz1);
        }

        float x = hsum_avx(xsums);
        float y = hsum_avx(ysums);
        float z = hsum_avx(zsums);

        return cv::Point3f(x, y, z);
    }

    // thirds: 0.0, 0.33, 0.66, 1.0, repeat
    // zs1: first two rows of 4 control point Z values
    // zs2: second two rows of 4 control point Z values
    // us: the coefficient products of u and u1
    // vs: the coefficient products of v and v1
    cv::Point3f evalBezierSurfaceCubicPointMatSimd3(const cv::Mat& controlPointsZs, float u, float v, __m256 thirds, __m256 zs1, __m256 zs2, __m256 us, __m256 vs)
    {
        __m256 xsums = _mm256_set1_ps(0.0f);
        __m256 ysums = _mm256_set1_ps(0.0f);
        __m256 zsums = _mm256_set1_ps(0.0f);

        // Unroll changed from 50 to 38 ms
        {
            int i1 = 0;
            int i2 = 1;

            __m256 bius = _mm256_permute_ps(us, 0x00);
            __m256 thirdsi = _mm256_set_ps(THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i2], THIRDS[i2], THIRDS[i2], THIRDS[i2]);

            __m256 p1 = _mm256_mul_ps(bius, vs);
            __m256 px1 = _mm256_mul_ps(p1, thirds);
            __m256 py1 = _mm256_mul_ps(p1, thirdsi);
            __m256 pz1 = _mm256_mul_ps(p1, zs1);

            xsums = _mm256_add_ps(xsums, px1);
            ysums = _mm256_add_ps(ysums, py1);
            zsums = _mm256_add_ps(zsums, pz1);
        }

        {
            int i1 = 2;
            int i2 = 3;

            __m256 bius = _mm256_permute_ps(us, 0b01010101);

            __m256 thirdsi = _mm256_set_ps(THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i1], THIRDS[i2], THIRDS[i2], THIRDS[i2], THIRDS[i2]);

            __m256 p1 = _mm256_mul_ps(bius, vs);
            __m256 px1 = _mm256_mul_ps(p1, thirds);
            __m256 py1 = _mm256_mul_ps(p1, thirdsi);
            __m256 pz1 = _mm256_mul_ps(p1, zs2);

            xsums = _mm256_add_ps(xsums, px1);
            ysums = _mm256_add_ps(ysums, py1);
            zsums = _mm256_add_ps(zsums, pz1);
        }

        float x = hsum_avx(xsums);
        float y = hsum_avx(ysums);
        float z = hsum_avx(zsums);

        return cv::Point3f(x, y, z);
    }
#endif

    // nPointsDim: each dimension, so total output points length nPointsDim^2
    // yStart: Index to start putting results into outputPoints. This will enlarge the outer (Y) vector if needed.
    // outputMat: cv::Point3f
    void evalBezierSurfaceCubicMat(const cv::Mat& controlPointsZs, int nPointsDim, float xOrigin, float yOrigin, cv::Mat& outputMat)
    {
        assert(controlPointsZs.rows == 4);
        assert(controlPointsZs.cols == 4);
        assert(outputMat.rows == nPointsDim);
        assert(outputMat.cols == nPointsDim);

        for (int yi = 0; yi < nPointsDim; yi++)
        {
            float v = (float)yi / nPointsDim;

            for (int xi = 0; xi < nPointsDim; xi++)
            {
                float u = (float)xi / nPointsDim;
                cv::Point3f p = evalBezierSurfaceCubicPointMat(controlPointsZs, v, u);
                //cv::Point3f p = evalBezierSurfaceCubicPointMatUnrolled(controlPointsZs, v, u);
                //cv::Point3f p = evalBezierSurfaceCubicPointMatUnrolled2(controlPointsZs, v, u);
                //cv::Point3f p = evalBezierSurfaceCubicPointMatSimd2(controlPointsZs, v, u);
                p.x += xOrigin;
                p.y += yOrigin;
                outputMat.at<cv::Point3f>(yi, xi) = p;
            }
        }
    }

#ifdef _WIN32
    // nPointsDim: each dimension, so total output points length nPointsDim^2
    // yStart: Index to start putting results into outputPoints. This will enlarge the outer (Y) vector if needed.
    // outputMat: cv::Point3f
    void evalBezierSurfaceCubicMatAvx(const cv::Mat& controlPointsZs, int nPointsDim, float xOrigin, float yOrigin, cv::Mat& outputMat)
    {
        assert(controlPointsZs.rows == 4);
        assert(controlPointsZs.cols == 4);
        assert(outputMat.rows == nPointsDim);
        assert(outputMat.cols == nPointsDim);

        // Preload some registers
        __m256 thirds = _mm256_set_ps(THIRDS[0], THIRDS[1], THIRDS[2], THIRDS[3], THIRDS[0], THIRDS[1], THIRDS[2], THIRDS[3]);

        const float* zi1 = controlPointsZs.ptr<float>(0, 0);
        const float* zi2 = controlPointsZs.ptr<float>(1, 0);
        __m256 zs1 = _mm256_set_ps(zi1[0], zi1[1], zi1[2], zi1[3], zi2[0], zi2[1], zi2[2], zi2[3]);

        const float* zi3 = controlPointsZs.ptr<float>(2, 0);
        const float* zi4 = controlPointsZs.ptr<float>(3, 0);
        __m256 zs2 = _mm256_set_ps(zi3[0], zi3[1], zi3[2], zi3[3], zi4[0], zi4[1], zi4[2], zi4[3]);

        // Pre-compute multiples of the u and v values.
        // Since we are computing on regularly spaced points, in both dimensions, we are re-using the same values.
        vector<float> fs(nPointsDim);
        vector<float> fs1(nPointsDim);
        vector<float> fs12(nPointsDim);
        vector<float> fs13(nPointsDim);
        vector<float> fs2(nPointsDim);
        vector<float> fs3(nPointsDim);

        for (int i = 0; i < nPointsDim; i++)
        {
            fs[i] = (float)i / nPointsDim;
            fs2[i] = fs[i] * fs[i];
            fs3[i] = fs2[i] * fs[i];

            fs1[i] = 1.0f - fs[i];
            fs12[i] = fs1[i] * fs1[i];
            fs13[i] = fs12[i] * fs1[i];
        }

        for (int yi = 0; yi < nPointsDim; yi++)
        {
            // Setup a u register with values where we can permute them out to another register where we need them.
            //float u = (float)yi / nPointsDim;
            //float u1 = 1.0f - u;
            //float biu1 = u1 * u1 * u1;
            //float biu2 = 3 * u * u1 * u1;
            //float biu3 = 3 * u * u * u1;
            //float biu4 = u * u * u;

            float u = fs[yi];
            float u1 = fs1[yi];
            float biu1 = fs13[yi];
            float biu2 = 3 * u * fs12[yi];
            float biu3 = 3 * fs2[yi] * u1;
            float biu4 = fs3[yi];

            //float us[4] = { biu1, biu2, biu3, biu4 };
            __m256 us = _mm256_set_ps(0.0f, 0.0f, biu3, biu1, 0.0f, 0.0f, biu4, biu2);

            for (int xi = 0; xi < nPointsDim; xi++)
            {
                //float v = (float)xi / nPointsDim;
                //float v1 = 1.0f - v;
                //float v13 = v1 * v1 * v1;
                //float v12 = 3 * v * v1 * v1;
                //float v21 = 3 * v * v * v1;
                //float v3 = v * v * v;

                float v = fs[xi];
                float v1 = fs1[xi];
                float v13 = fs13[xi];
                float v12 = 3 * v * fs12[xi];
                float v21 = 3 * fs2[xi] * v1;
                float v3 = fs3[xi];

                __m256 vs = _mm256_set_ps(v13, v12, v21, v3, v13, v12, v21, v3);

                //cv::Point3f p = evalBezierSurfaceCubicPointMat(controlPointsZs, v, u);
                //cv::Point3f p = evalBezierSurfaceCubicPointMatUnrolled(controlPointsZs, v, u);
                //cv::Point3f p = evalBezierSurfaceCubicPointMatUnrolled2(controlPointsZs, v, u);
                //cv::Point3f p = evalBezierSurfaceCubicPointMatSimd2(controlPointsZs, v, u);
                cv::Point3f p = evalBezierSurfaceCubicPointMatSimd3(controlPointsZs, u, v, thirds, zs1, zs2, us, vs);
                p.x += xOrigin;
                p.y += yOrigin;
                outputMat.at<cv::Point3f>(yi, xi) = p;
            }
        }
    }
    #endif
}
