package chowClustering;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class AIC_test {

	public static void main(String[] args) {
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("C:/Users/user/git/spawnn/output/chicago.shp"), true);
		int[] fa = new int[]{3,7,9,10};
		int ta = 6;
		
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden_gs2010/gem_dat.shp"), new int[]{ 1, 2 }, true);
		int[] fa = new int[] { 7,  8,  9, 10, 19, 20 };
		int ta = 18; // bldUpRt*/
						
		List<Set<double[]>> cluster = new ArrayList<>();
		cluster.add( new HashSet<>(sdf.samples));
		
		{
			PiecewiseLM plm = new PiecewiseLM(sdf.samples, cluster, "", fa, ta, false);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.betas.get(0)));
			System.out.println(plm.getAICc()+", "+plm.getRSS());
		}
		
		System.out.println("----");
		
		{
			PiecewiseLM plm = new PiecewiseLM(sdf.samples, cluster, "", fa, ta, true);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.betas.get(0)));
			System.out.println(plm.getAICc()+", "+plm.getRSS());
		}
				
		{
			PiecewiseLM plm = new PiecewiseLM(sdf.samples, cluster, "", fa, ta, true, 0.01);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.betas.get(0)));
			System.out.println(plm.getAICc()+", "+plm.getRSS());
		}	
	}
}
