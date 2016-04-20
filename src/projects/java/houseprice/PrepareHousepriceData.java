package houseprice;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class PrepareHousepriceData {

	private static Logger log = Logger.getLogger(PrepareHousepriceData.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);
				
		List<double[]> samples = sdf.samples;
		log.debug(samples.size());
		
		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");

		//0.842800042346674,0.13755004180349054,1.3265852052352125E-74
		vars.add("lnarea_tot"); //0.913923245395007,0.14380591197252182,1.79615169870852E-81
		
		vars.add("lnarea_plo"); //0.9997901553930167,0.13909227915923866,3.9895058992366924E-76
		vars.add("attic_dum"); //0.996167146077074,0.14143660897664934,3.123858411374094E-79
		vars.add("cellar_dum"); //0.9851226796055021,0.13581156736405078,6.022218800523935E-73
		vars.add("cond_house_3"); //0.9944282636883391,0.14051955925189347,2.7422167483332165E-78
		vars.add("heat_3"); //0.9734988292582646,0.14023839424403745,1.4383195843856763E-77
		vars.add("bath_3"); //0.9797904046427245,0.13907101093325486,1.4477674393280417E-76
		vars.add("garage_3"); //0.9913299816404059,0.14288683534484026,1.688154732952732E-80
		vars.add("terr_dum"); //0.9840335531704287,0.1319615754198577,9.568249344699974E-69
		vars.add("age_num"); //0.9474322059466647,0.13867248951877562,2.1134350740490176E-76					
		
		//vars.add("time_index"); //0.9998715224265088,0.13845002844913828,2.9411508335018544E-76
		
		// Bivand: No contextual variables about the neighbourhood of the houses are available, so one would expect a strong spatial autocorrelation reflecting this misspecification.
		vars.add("zsp_alq_09");
		vars.add("gem_kauf_i");
		vars.add("gem_abi");
		vars.add("gem_alter_");
		vars.add("ln_gem_dic");
		
		vars.add("lnp");
		
		List<double[]> ns = new ArrayList<double[]>();
		for( double[] d : samples ) {
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			ns.add(nd);
		}
		
		DataUtils.writeCSV("output/houseprice.csv", ns, vars.toArray(new String[]{}));	
	}
}
