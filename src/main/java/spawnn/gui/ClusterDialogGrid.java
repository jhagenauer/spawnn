package spawnn.gui;

import java.awt.CardLayout;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class ClusterDialogGrid extends JDialog implements ActionListener {

	JPanel cards;
	JComboBox cb;

	JTextField numCluster_1, numCluster_2, min, max, blur;
	JCheckBox connected;

	JButton ok, cancel;
	boolean okPressed = false;

	public enum ClusterAlgorithm {
		kMeans, Ward, SLK, ALK, CLK, SKATER, Watershed
	};

	private enum Card {
		one, two, three
	};

	private static final long serialVersionUID = -6577606485481195079L;

	public ClusterDialogGrid(Frame parent, String string, boolean b, boolean enableWatershed ) {
		super(parent, string, b);

		setLayout(new MigLayout(""));

		add(new JLabel("Algorithm:"));
		cb = new JComboBox();
		cb.setModel(new DefaultComboBoxModel(ClusterAlgorithm.values()));
		cb.addActionListener(this);
		if( !enableWatershed )
			cb.removeItem(ClusterAlgorithm.Watershed);
		add(cb, "wrap");

		// no contiguity option
		JPanel jp_1 = new JPanel(new MigLayout());
		jp_1.add(new JLabel("Number of clusters:"));
		numCluster_1 = new JTextField("5", 3);
		jp_1.add(numCluster_1, "wrap");

		// with contiguity option
		JPanel jp_2 = new JPanel(new MigLayout());
		jp_2.add(new JLabel("Number of clusters:"));
		numCluster_2 = new JTextField("5", 3);
		jp_2.add(numCluster_2, "wrap");

		jp_2.add(new JLabel("Contiguity constrained?"));
		connected = new JCheckBox();
		jp_2.add(connected, "wrap");

		// watershed
		JPanel jp_3 = new JPanel(new MigLayout());
		jp_3.add(new JLabel("Minimum:"));
		min = new JTextField("55", 3);
		jp_3.add(min, "wrap");
		jp_3.add(new JLabel("Maximum:"));
		max = new JTextField("255", 3);
		jp_3.add(max, "wrap");
		jp_3.add(new JLabel("Blur:"));
		blur = new JTextField("8.0", 3);
		jp_3.add(blur, "wrap");

		cards = new JPanel(new CardLayout());
		cards.add(jp_1, Card.one.toString());
		cards.add(jp_2, Card.two.toString());
		cards.add(jp_3, Card.three.toString());

		add(cards, "span 2, wrap");

		ok = new JButton("OK");
		ok.addActionListener(this);
		cancel = new JButton("Cancel");
		cancel.addActionListener(this);
		add(ok, "split 2, push");
		add(cancel);

		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		pack();
		setAlwaysOnTop(true);
		setLocationRelativeTo(parent);
		setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == cb) {
			CardLayout cl = (CardLayout) (cards.getLayout());
			if (cb.getSelectedItem() == ClusterAlgorithm.Watershed)
				cl.show(cards, Card.three.toString());
			else if (cb.getSelectedItem() == ClusterAlgorithm.SKATER || cb.getSelectedItem() == ClusterAlgorithm.kMeans)
				cl.show(cards, Card.one.toString());
			else
				cl.show(cards, Card.two.toString());

		} else if (e.getSource() == ok) {
			okPressed = true;
			dispose();
		} else {
			okPressed = false;
			dispose();
		}
	}

	public boolean isOkPressed() {
		return okPressed;
	}

	public ClusterAlgorithm getAlgorithm() {
		return (ClusterAlgorithm) cb.getSelectedItem();
	}

	public int getNumCluster() {
		if (cb.getSelectedItem() == ClusterAlgorithm.SKATER || cb.getSelectedItem() == ClusterAlgorithm.kMeans)
			return Integer.parseInt(numCluster_1.getText());
		else
			return Integer.parseInt(numCluster_2.getText());
	}

	public boolean getConnected() {
		return connected.isSelected() || cb.getSelectedItem() == ClusterAlgorithm.SKATER;
	}
	
	public int getMinimum() {
		return Integer.parseInt(min.getText());
	}
	
	public int getMaximum() {
		return Integer.parseInt(max.getText());
	}
	
	public double getBlur() {
		return Double.parseDouble(blur.getText());
	}
}
