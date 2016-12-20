package chowClustering;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class AIC_test {

	public static void main(String[] args) {
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("C:/Users/user/git/spawnn/output/chicago.shp"), true);
		int[] fa = new int[]{3,7,9,10};
		int[] ga = new int[]{0,1};
		int ta = 6;*/
		
		//SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden_gs2010/gem_dat.shp"), new int[]{ 1, 2 }, true);
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/Export_Output.shp"), new int[]{ 1, 2 }, true);
		int[] fa = new int[] { 7,  8,  9, 10, 19, 20 };
		int[] ga = new int[]{ 3, 4 };
		int ta = 18; // bldUpRt

		List<Set<double[]>> cluster = new ArrayList<>(Clustering.kMeans(sdf.samples, 1, new EuclideanDist(ga)).values());
		
		{
			long time = System.currentTimeMillis();
			PiecewiseLM plm = new PiecewiseLM(sdf.samples, cluster, "", fa, ta, false);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.getBeta(0)));
			System.out.println(plm.getR2(sdf.samples));
			System.out.println("took: "+(System.currentTimeMillis()-time)/1000.0);
		}
		
		{
			long time = System.currentTimeMillis();
			PiecewiseLM plm = new PiecewiseLM(sdf.samples, cluster, "", fa, ta, true, 0.0);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.getBeta(0)));
			System.out.println( plm.getR2(sdf.samples));
			System.out.println("took: "+(System.currentTimeMillis()-time)/1000.0);
		}
		
		{
			long time = System.currentTimeMillis();
			PiecewiseLM plm = new PiecewiseLM(sdf.samples, cluster, "", fa, ta, true, 0.01);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.getBeta(0)));
			System.out.println( plm.getR2(sdf.samples));
			System.out.println("took: "+(System.currentTimeMillis()-time)/1000.0);
		}
	}
}
