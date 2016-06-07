package regionalization.medoid;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class PrepareCensusData {

	public static void main(String[] args) {

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/census2010/sf1/DEC_10_SF1_SF1DP1_with_ann.csv"), new int[]{}, false);
		for( int i = 0; i < df.names.size(); i++ )
			System.out.println(i+","+df.names.get(i)+","+i);
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/census2010/boundary/gz_2010_us_050_00_500k.shp"), true);
		Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(sdf.samples, sdf.geoms,false);
		List<Map<double[], Set<double[]>>> sub = GraphUtils.getSubGraphs(cm);

		Map<double[], Set<double[]>> largestSub = null;
		for (Map<double[], Set<double[]>> s : sub) {
			if (largestSub == null || largestSub.size() < s.size())
				largestSub = s;
		}

		try {
			CoordinateReferenceSystem targetCRS = CRS.decode("EPSG:102008");
			MathTransform transform = CRS.findMathTransform(sdf.crs, targetCRS, true); // requires epsg-extensions-plugin
			List<double[]> nSamples = new ArrayList<double[]>();
			List<Geometry> nGeoms = new ArrayList<Geometry>();
			for (double[] n : GraphUtils.getNodes(largestSub)) {
				int state = (int) n[1];
				int county = (int) n[2];

				for (double[] d : df.samples) {
					String id = (int) d[0] + "";
					int s = Integer.parseInt(id.substring(0, id.length() - 3)); // first digits
					int c = Integer.parseInt(id.substring(id.length() - 3, id.length())); // last 3 digits

					if (state != s || county != c)
						continue;
					
					int idx = sdf.samples.indexOf(n);
					Geometry g = JTS.transform(sdf.geoms.get(idx), transform);
					nGeoms.add(g);
					
					double totPop = d[49]+d[98];
					double[] nd = new double[] {
							g.getCentroid().getX(),
							g.getCentroid().getY(),
							(d[2]+d[4]+d[6]+d[8]+d[10])/totPop, // 0 to 24
							(d[12]+d[14]+d[16]+d[18]+d[20]+d[22]+d[24]+d[26])/totPop, // 25 to 64
							(d[28]+d[30]+d[32]+d[34]+d[36])/totPop, // 65 older
							//d[38], // median age
							d[151], // white
							d[153], // black
							d[155], // indian
							d[157], // asian
							//d[173], // hawaiian
							//d[183], // other
							(d[172]+d[182])/totPop, // hawaiian + other
							d[185], // 2 races
							d[208], // hispanic
							d[253], // in households
							d[291], // family households
							d[322], // avg household size
							//d[323], // avg family size
							d[347], // renter-occupied housing units
							//d[349], // avg household size of renter-occupied housing units		
					};
										
					nSamples.add(nd);
				}
			}		
			String[] names = new String[]{
					"x",
					"y",
					"0to24",
					"25to64",
					"65older",
					//"mAge",
					"white",
					"black",
					"indian",
					"asian",
					"other",
					"2races",
					"hispanic",
					"inHH",
					"famHH",
					"avgHHs",
					//"avgFams",
					"rentOcc",
					//"rntOcHHs"
			};
						
			DataUtils.writeShape(nSamples, nGeoms, names, targetCRS, "output/counties.shp");
			DataUtils.writeCSV("output/counties.csv", nSamples, names);
		} catch (MismatchedDimensionException e) {
			e.printStackTrace();
		} catch (TransformException e) {
			e.printStackTrace();
		} catch (NoSuchAuthorityCodeException e) {
			e.printStackTrace();
		} catch (FactoryException e) {
			e.printStackTrace();
		} finally {
		}
	}
}
