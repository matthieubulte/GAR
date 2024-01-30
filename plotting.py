import matplotlib.pyplot as plt
import numpy as np

def compute_critical_val(df, quantiles_field, alpha):
    computed_alphas = np.linspace(0, 1, 50)
    def _comp(qs): return np.interp(alpha, computed_alphas, qs)
    return df[quantiles_field].apply(_comp)

def plot_test_for(df, statname, onesided=False):
    alpha = 0.05

    if onesided:
        crit = compute_critical_val(df, f'quantiles_{statname}', 1-alpha)
        df['rejected'] = df[statname] > crit
    else:
        crit_lo = compute_critical_val(df, f'quantiles_{statname}', alpha/2)
        crit_hi = compute_critical_val(df, f'quantiles_{statname}', 1 - alpha/2)
        df['rejected'] = (df[statname] < crit_lo) | (df[statname] > crit_hi)

    results = df.groupby(['sample_size', 'phi']).agg(rejection_rate=('rejected', 'mean')).reset_index()

    for phi in results.phi.unique():
        sub_df = results[results['phi'] == phi]

        plt.scatter(sub_df['sample_size'], sub_df.rejection_rate, s=5)
        if phi == 0:
            plt.plot(sub_df['sample_size'], sub_df.rejection_rate, label=r'$0\ (H_0)$')
        else:
            plt.plot(sub_df['sample_size'], sub_df.rejection_rate, label=np.round(phi,1))
    plt.xlabel('Sample Size')
    plt.ylabel('Rejection Rate')
    plt.legend(title=r'$\varphi$')
    plt.grid('on')
    plt.yticks([0.05,0.1,0.2,0.3,0.4,0.5,0.6,.7,.8,.9,1]);

def plot_alpha_alpha(df, statname, samplesize, onesided=False):
    target_sizes = np.linspace(0, .1, 40)
    for phi in np.sort(df.phi.unique()):
        sub_df = df[(df['sample_size'] == samplesize) & (df['phi'] == phi)]

        emp_sizes = np.zeros_like(target_sizes)
        for i in range(target_sizes.shape[0]):
            alpha=target_sizes[i]

            if onesided:
                crit = compute_critical_val(sub_df, f'quantiles_{statname}', 1-alpha)
                emp_sizes[i] = (sub_df[statname] > crit).mean()
            else:
                crit_lo = compute_critical_val(sub_df, f'quantiles_{statname}', alpha/2)
                crit_hi = compute_critical_val(sub_df, f'quantiles_{statname}', 1 - alpha/2)
                emp_sizes[i] = ((sub_df[statname] < crit_lo) | (sub_df[statname] > crit_hi)).mean()
            
        label = r'$0\ (H_0)$' if phi == 0 else f'{phi:.1f}'
        # plt.scatter(target_sizes, emp_sizes, label=label, s=5)
        plt.plot(target_sizes, emp_sizes, label=label, marker='o', markersize=3)

    plt.legend(title=r'$\varphi$')
    plt.xlabel('Test level')
    plt.ylabel('Rejection Rate')
    plt.ylim(0,1)
    plt.grid('on')
