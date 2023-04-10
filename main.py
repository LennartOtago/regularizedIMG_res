import numpy as np
from numpy.fft import fft2, ifft2, fft
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lcurve_functions import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import numpy.random as rd


def direct_linear_2d(x,h):
    r = np.zeros((len(x),len(x)))
    for k in range(len(x) ):
        for l in range(len(x)):

            for m in range(len(x)):
                for n in range(len(x)):
                    if k-m >= 0 and (l-n)>= 0 and (k-m) < len(h) and (l-n) < len(h) :
                        r[k,l] = r[k,l] + (x[m,n] * h[(k-m),(l-n)])
    return r


#2D circular convolution
#matrices have to be same size and conv_matrix has to be zero padded
def direct_circular_2d(x,h):
    r = np.zeros((len(x),len(x)))
    for k in range(len(x)):
        for l in range(len(x)):

            for m in range(len(x)):
                for n in range(len(x)):
                    #print('k-p ' + str((k - p)))
                    #print((k - p) % len(g))
                    r[k,l] = r[k,l] + (x[m,n] * h[ (k-m) % (len(h)) , (l-n) % (len(h)) ])

    return r



def transpose_conv_2d(X,H, padding):
#H is kernel
#X and H have to be squared
#padding reduces the size of the output equally
    X_pad = np.pad(X, (0,padding))
    for x in range(padding):
        X_pad[len(X)+x,:] = X_pad[x,:]
        X_pad[:, len(X) + x] = X_pad[:,x]
        X_pad[len(X) + x, len(X) + x] =  X_pad[x,x]
    output_size = len(X_pad) - len(H)
    Z = np.zeros((output_size,output_size))
    #pad X

#stride
    k = 0

    for i in range(output_size):
        l = 0
        for j in range(output_size):
            Z[k,l] =  sum(sum(X_pad[i: (i+len(H)),j: (j+len(H)) ] * H))
            l = l + 1
        k = k + 1

    return Z

# ################################################################


#####################################
gray_img = mpimg.imread('jupiter1.tif')


# get psf from satellite
org_img = np.array(gray_img)#/sum(sum(np.array(gray_img)))
four_conv = fft2(org_img)#, axes = (1,0) ) #/ np.sqrt(256**2)
# plt.imshow(org_img, cmap='gray')
# plt.show()



xpos = 234
ypos = 85  # Pixel at centre of satellite
sat_img_org = org_img[ypos - 16: ypos + 16, xpos - 16:xpos + 16]
sat_img = sat_img_org / (sum(sum(sat_img_org)))
sat_img[sat_img < 0.05*np.max(sat_img)] = 0

# plt.imshow(sat_img, cmap='gray')
# plt.show()


pad_img = np.zeros((256,256))
pad_img[0:32,0:32] = sat_img
fourier_img = fft2(sat_img, (256,256))#, axes =(1,0) )
fourier_img2 = fft2(pad_img)
# # naive inversion by direct division
print(np.allclose(fourier_img, fourier_img2, atol= 1e-10))





lam = 0.100#19

tikh_img_store = ( ifft2( four_conv * np.conj(fourier_img) / (lam ** 2 +  abs(fourier_img)**2 )))
tikh_img = tikh_img_store.real#[0:256,0:256]

Z_fft = ifft2(four_conv * np.conj(fourier_img)).real
Z =  transpose_conv_2d(org_img, sat_img, len(sat_img))
print(np.allclose(Z, Z_fft, atol= 1e-9))


alpha = 0.0056724
L_org = np.array([[ 0, -1, 0],[ -1, 4, -1],[ 0, -1, 0]])
four_L = fft2(  L_org, (256,256))


print('bla')


# alphas = np.linspace(0.000001,0.1,1000)
# #alphas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) * 1e-3
alphas = np.logspace(-10, 1, 200) #np.linspace(0,1e-6, 200)#
norm_f = np.zeros(len(alphas))
norm_data = np.zeros(len(alphas))


for i in range(0, len(alphas)):

    reg_img = (four_conv * np.conj(fourier_img)/ (alphas[i] * abs(four_L) +  abs(fourier_img)**2))
    #x = np. matrix.flatten(reg_img)
    # norm parseval theorem
    # take real part.. think about imaginary part as noise as x.conj * x should be real valued
    norm_f[i] = np.sqrt( sum(sum( (reg_img.conj() * abs(four_L) * reg_img).real )))/256

    norm_data[i] = np.linalg.norm(ifft2(four_conv - reg_img * fourier_img))

reg_img = (four_conv * np.conj(fourier_img) / (alphas[109] * abs(four_L) + abs(fourier_img) ** 2))
# take real value of img as we have real input and need real output
c = (ifft2(reg_img)).real
plt.imshow(c, cmap='gray')
plt.show()


fig4 = plt.figure()
ax4 = fig4.add_subplot()
ax4.set_xscale('log')
ax4.set_yscale('log')
plt.plot(alphas,norm_data)

fig5 = plt.figure()
ax5 = fig5.add_subplot()
ax5.set_xscale('log')
ax5.set_yscale('log')
plt.plot(alphas,norm_f)

plt.show()




fig2 = plt.figure()
ax = fig2.add_subplot()
ax.set_xscale('log')
ax.set_yscale('log')
# plt.xlim((10,1e4))
# plt.ylim((1e3,1e6))
# print(lambas[::100])

plt.plot(norm_data, norm_f,'o', color='black')

#plot crosses from MTC
MTCnorms= np.loadtxt('norms.txt')
plt.plot(MTCnorms[:,0], MTCnorms[:,1],  marker = "." ,mfc = 'black' , markeredgecolor='r',linestyle = 'None')
#k = 0
axins = zoomed_inset_axes(ax,30,loc='lower left')
axins.plot(norm_data, norm_f,'o', color='black')
axins.plot(MTCnorms[:,0], MTCnorms[:,1], marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')
#axins.scatter(norm_data, norm_f)

x1, x2, y1, y2 = 5.4e2, 5.6e2, 3.5e4, 3.7e4 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
axins.tick_params( bottom=False, left=False, labelbottom = False, labelleft = False)

mark_inset(ax, axins, loc1=1, loc2=4 ,fc="none", ec="0.5")
#from l_curve in matlab
k=134
txt = alphas[k]
ax.annotate(np.around(txt,6), (norm_data[k]+15, norm_f[k]+15))
ax.plot(norm_data[k], norm_f[k],  marker = "." ,mfc = 'lime' , markeredgecolor='green',linestyle = 'None',markersize=10)
# i = 109
# txt = alphas[i]
# ax.annotate(np.around(txt,5), (norm_data[i], norm_f[i]))
plt.show()


#plot posterior
minimizer = norm_data**2 + alphas *  norm_f**2
fig3 = plt.figure()
ax = fig3.add_subplot()
#ax.set_xscale('log')
#ax.set_yscale('log')
plt.plot(alphas, minimizer )
plt.show()


print("bla")