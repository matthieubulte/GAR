import matplotlib.pyplot as plt
import numpy as np

def compute_critical_val(df, quantiles_field, alpha):
    computed_alphas = np.linspace(0, 1, 50)
    def _comp(qs): return np.interp(alpha, computed_alphas, qs)
    return df[quantiles_field].apply(_comp)

def plot_sqrt_T_mu(df):
    for phi in np.sort(df.phi.unique()):
        if phi == 1:
            continue
        sub_df = df[df['phi'] == phi].copy()
        grped = sub_df.groupby('sample_size').mean(numeric_only=True).reset_index()
        label = r'$0\ (H_0)$' if phi == 0 else f'{phi:.1f}'
        plt.plot(grped['sample_size'], np.sqrt(grped['sample_size']*grped['err']), label=label, marker='o', markersize=4)

    plt.grid(True)
    plt.xlabel('Sample Size')
    plt.xlim(left=0)
    plt.ylabel(r'$\sqrt{T}d(\mu, \hat\mu_T)$')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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

        label = r'$0\ (H_0)$' if phi == 0 else np.round(phi,1)
        plt.plot(sub_df['sample_size'], sub_df.rejection_rate, label=label, marker='o', markersize=4)#, linewidth=0.75)
        # plt.scatter(, s=5)
        # if phi == 0:
        #     plt.plot(sub_df['sample_size'], sub_df.rejection_rate, label=)
        # else:
        #     plt.plot(sub_df['sample_size'], sub_df.rejection_rate, label=label)
    plt.xlabel('Sample Size')
    plt.ylabel('Rejection Rate (%)')
    # plt.legend(title=r'$\varphi$', ncols=results.phi.nunique())
    plt.grid(True)
    plt.xlim(left=0)
    plt.yticks([0.05,0.1,0.2,0.3,0.4,0.5,0.6,.7,.8,.9,1], [5,10,20,30,40,50,60,70,80,90,100]);
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_alpha_alpha(df, statname, samplesize, onesided=False):
    target_sizes = np.linspace(0, .1, 5)
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
        plt.plot(target_sizes, emp_sizes, label=label, marker='o', markersize=4)#, linewidth=0.75)

    # plt.legend(title=r'$\varphi$')
    plt.plot(target_sizes, target_sizes, color='black', linestyle='--')#, linewidth=0.75)
    plt.xlabel('Test level (%)')
    plt.ylabel('Rejection Rate (%)')
    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,.7,.8,.9,1], [10,20,30,40,50,60,70,80,90,100]);
    plt.xticks([0, 0.02, 0.04, 0.06, 0.08 , 0.1], [0, 2, 4, 6, 8, 10]);
    plt.xlim(0, target_sizes[-1])
    plt.ylim(0,1.05)
    plt.grid(True)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
